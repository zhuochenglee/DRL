import time

import numpy as np
from stable_baselines3 import SAC

from environment_3 import TopologicalDynamicEnv, TrainingProgressBarCallback

N_TEST_SCENARIOS = 10
RL_TRAINING_STEPS = 200000

# 历史经验范围 (规则调度使用，不依赖当前场景的未来信息)
HIST_DEMAND_MIN = 110.0
HIST_DEMAND_MAX = 280.0

# 历史基准需求 (GA/DP 规划用，不含实际噪声)
BASE_DEMAND = np.array([
    120, 110, 110, 115, 120, 130, 150, 180,
    220, 250, 260, 250, 240, 240, 250, 260,
    270, 280, 280, 260, 220, 180, 150, 130
], dtype=np.float64)


# ==========================================
# 场景生成
# ==========================================
def generate_test_scenarios(n, seed=123):
    """生成 N 个随机测试场景 (需求曲线 + 电价曲线)"""
    rng = np.random.default_rng(seed)
    base_demand = np.array([
        120, 110, 110, 115, 120, 130, 150, 180,
        220, 250, 260, 250, 240, 240, 250, 260,
        270, 280, 280, 260, 220, 180, 150, 130
    ], dtype=np.float64)
    base_prices = np.array([
        0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
        1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
        1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.3, 0.3
    ], dtype=np.float64)
    scenarios = []
    for _ in range(n):
        noise = rng.normal(0, 0.15, size=24)
        demand = np.clip(base_demand * (1.0 + noise), 50, 400)
        shift = int(rng.integers(-2, 3))
        prices = np.roll(base_prices, shift)
        scenarios.append((demand.copy(), prices.copy()))
    return scenarios


# ==========================================
# 通用单 episode 评估
# ==========================================
def run_episode(env, policy_fn, demand, prices):
    """在指定场景上跑 24h，policy_fn(obs, env, hour) -> action array"""
    obs, _ = env.reset(options={"demand": demand, "prices": prices})
    total_power = 0.0
    total_cost = 0.0
    violation_count = 0
    records = []

    for h in range(24):
        action = policy_fn(obs, env, h)
        obs, reward, terminated, truncated, info = env.step(action)
        total_power += info["power"]
        total_cost += info["power"] * prices[h]
        if info["violation"]:
            violation_count += 1
        records.append({
            "hour": h,
            "price": float(prices[h]),
            "demand": float(demand[h]),
            "alpha": float(action[0]),
            "p2_internal": info["p2_internal"],
            "p3_user": info["p3_user"],
            "power": info["power"],
        })
        if terminated:
            break

    return {
        "total_power": float(total_power),
        "total_cost": float(total_cost),
        "violation_count": int(violation_count),
        "records": records,
    }


# ==========================================
# 方法一：传统规则调度 (当前需求 + 已知电价序列)
# ==========================================
def rule_policy(obs, env, hour):
    demand = env._episode_demand[hour]
    price = env._episode_prices[hour]
    prices = env._episode_prices  # 完整电价序列 (已知)
    p2, p3 = env.P_internal

    ratio = np.clip((demand - HIST_DEMAND_MIN) / (HIST_DEMAND_MAX - HIST_DEMAND_MIN), 0, 1)
    alpha = 1.05 + 0.55 * ratio

    if p3 < env.p_min_safe + 5.0:
        alpha += 0.25
    elif p3 > env.p_min_safe + 20.0:
        alpha -= 0.08
    if p2 < env.p_min_safe + 10.0:
        alpha += 0.12

    # 电价前瞻: 利用已知的未来电价做缓冲管理
    if hour < 23:
        next_price = prices[hour + 1]
        if price <= 0.5 and next_price > 0.5:
            alpha += 0.10  # 谷转峰前多注气，建立压力缓冲
        elif price > 0.5 and next_price <= 0.5:
            alpha -= 0.05  # 峰转谷前少注气，等便宜电

    if price > 0.5 and p3 > env.p_min_safe + 10.0:
        alpha -= 0.10
    elif price <= 0.5 and p3 < env.p_min_safe + 15.0:
        alpha += 0.08

    return np.array([np.clip(alpha, 1.0, 2.0)], dtype=np.float32)


# ==========================================
# 方法二：遗传算法 (已知电价 + 基准需求预测的离线优化)
# ==========================================
def ga_optimize_scenario(demand, prices, pop_size=60, gens=80, elite=6,
                         mut_rate=0.12, mut_scale=0.08, rng=None, C=None):
    if rng is None:
        rng = np.random.default_rng(42)

    eval_env = TopologicalDynamicEnv(noise_scale=0.0, C=C)

    def eval_schedule(schedule):
        def policy(obs, env, hour):
            return np.array([schedule[hour]], dtype=np.float32)
        return run_episode(eval_env, policy, demand, prices)

    # 用规则策略生成一个启发式种子个体
    heuristic = []
    obs, _ = eval_env.reset(options={"demand": demand, "prices": prices})
    for h in range(24):
        a = rule_policy(obs, eval_env, h)
        heuristic.append(float(a[0]))
        if h < 23:
            obs, _, _, _, _ = eval_env.step(a)

    pop = rng.uniform(1.0, 2.0, size=(pop_size, 24))
    pop[0] = np.array(heuristic)

    best_schedule = pop[0].copy()
    best_cost = float("inf")

    for gen in range(gens):
        results = [eval_schedule(ind) for ind in pop]
        costs = np.array([r["total_cost"] + r["violation_count"] * 100000 for r in results])
        fitness = -costs
        ranked = np.argsort(fitness)[::-1]

        top_result = results[ranked[0]]
        top_real_cost = top_result["total_cost"]
        if top_result["violation_count"] == 0 and top_real_cost < best_cost:
            best_schedule = pop[ranked[0]].copy()
            best_cost = top_real_cost
        elif best_cost == float("inf"):
            best_schedule = pop[ranked[0]].copy()
            best_cost = top_real_cost

        next_pop = [pop[i].copy() for i in ranked[:elite]]
        while len(next_pop) < pop_size:
            idxs = rng.integers(0, pop_size, size=4)
            pa = pop[idxs[np.argmax(fitness[idxs])]]
            idxs = rng.integers(0, pop_size, size=4)
            pb = pop[idxs[np.argmax(fitness[idxs])]]
            blend = rng.uniform(0, 1, size=24)
            child = blend * pa + (1 - blend) * pb
            mask = rng.random(24) < mut_rate
            child += mask * rng.normal(0, mut_scale, size=24)
            child = np.clip(child, 1.0, 2.0)
            next_pop.append(child)
        pop = np.array(next_pop, dtype=np.float32)

    return best_schedule


# ==========================================
# 方法三：动态规划 (已知电价 + 基准需求预测的离线优化)
# ==========================================
def dp_optimize_scenario(demand, prices, n_p=30, n_a=25, C=None):
    """离散状态空间动态规划，基于需求预测+已知电价反向递推"""
    K = np.array([0.02, 0.05])
    C = np.array(C, dtype=np.float64) if C is not None else np.array([0.05, 0.05])
    p0_ref = 100.0
    p_min_safe = 80.0
    p_max_safe = 150.0

    p_grid = np.linspace(10.0, 200.0, n_p)
    a_grid = np.linspace(1.0, 2.0, n_a)
    dp_val = p_grid[1] - p_grid[0]

    # 3D 广播: (n_p, 1, 1), (1, n_p, 1), (1, 1, n_a)
    P2 = p_grid[:, None, None]
    P3 = p_grid[None, :, None]
    ALPHA = a_grid[None, None, :]
    P_SRC = p0_ref * ALPHA

    V_next = np.zeros((n_p, n_p))
    best_action = np.zeros((24, n_p, n_p), dtype=int)

    for hour in range(23, -1, -1):
        d = demand[hour]
        price = prices[hour]

        # 批量计算物理转移
        dP0 = P_SRC**2 - P2**2
        dP1 = P2**2 - P3**2
        Q0 = np.sign(dP0) * np.sqrt(np.abs(dP0) / K[0])
        Q1 = np.sign(dP1) * np.sqrt(np.abs(dP1) / K[1])

        nxt_p2 = np.clip(P2 + C[0] * (Q0 - Q1), 10.0, 200.0)
        nxt_p3 = np.clip(P3 + C[1] * (Q1 - d), 10.0, 200.0)

        power = np.abs(Q0) * (ALPHA - 1.0)

        # 越限惩罚
        viol = (
            np.where(nxt_p2 < p_min_safe, 1000 * (p_min_safe - nxt_p2), 0)
            + np.where(nxt_p2 > p_max_safe, 1000 * (nxt_p2 - p_max_safe), 0)
            + np.where(nxt_p3 < p_min_safe, 1000 * (p_min_safe - nxt_p3), 0)
            + np.where(nxt_p3 > p_max_safe, 1000 * (nxt_p3 - p_max_safe), 0)
        )

        stage_cost = power * price + viol

        # 双线性插值查 V_next
        idx2 = np.clip((nxt_p2 - p_grid[0]) / dp_val, 0, n_p - 1.001)
        idx3 = np.clip((nxt_p3 - p_grid[0]) / dp_val, 0, n_p - 1.001)
        i2 = np.clip(idx2.astype(int), 0, n_p - 2)
        i3 = np.clip(idx3.astype(int), 0, n_p - 2)
        f2 = idx2 - i2
        f3 = idx3 - i3

        future = (
            (1 - f2) * (1 - f3) * V_next[i2, i3]
            + f2 * (1 - f3) * V_next[i2 + 1, i3]
            + (1 - f2) * f3 * V_next[i2, i3 + 1]
            + f2 * f3 * V_next[i2 + 1, i3 + 1]
        )

        total = stage_cost + future
        best_action[hour] = np.argmin(total, axis=2)
        V_next = np.min(total, axis=2)

    # 前向模拟: 从初始状态 (100, 100) 沿最优策略走
    schedule = []
    p2, p3 = 100.0, 100.0
    for hour in range(24):
        i2 = int(np.clip(round((p2 - p_grid[0]) / dp_val), 0, n_p - 1))
        i3 = int(np.clip(round((p3 - p_grid[0]) / dp_val), 0, n_p - 1))
        alpha = a_grid[best_action[hour, i2, i3]]
        schedule.append(float(alpha))

        # 精确物理前推
        dP0 = (p0_ref * alpha) ** 2 - p2**2
        dP1 = p2**2 - p3**2
        q0 = np.sign(dP0) * np.sqrt(abs(dP0) / K[0])
        q1 = np.sign(dP1) * np.sqrt(abs(dP1) / K[1])
        p2 = float(np.clip(p2 + C[0] * (q0 - q1), 10.0, 200.0))
        p3 = float(np.clip(p3 + C[1] * (q1 - demand[hour]), 10.0, 200.0))

    return np.array(schedule)


# ==========================================
# 方法四：强化学习策略 (仅基于当前观测)
# ==========================================
def make_rl_policy(model):
    def policy(obs, env, hour):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


# ==========================================
# 打印工具
# ==========================================
def print_episode_table(result):
    print(f"\n{'时间':<6} | {'电价':<4} | {'需求':<6} | {'动作(排比)':<8} | {'节点2压':<7} | {'节点3压':<7} | {'总能耗(kW)':<8}")
    print("-" * 75)
    for r in result["records"]:
        print(
            f"{r['hour']:02d}:00  | {r['price']:.1f}元 | {r['demand']:<6.0f} | "
            f"{r['alpha']:<8.4f} | {r['p2_internal']:<7.1f} | {r['p3_user']:<7.1f} | {r['power']:<8.1f}"
        )
    print("-" * 75)
    print(f"\n24 小时运行总结报告:")
    print(f"-> 全天总耗电量: {result['total_power']:.2f} kWh")
    print(f"-> 全天总运行成本: ¥ {result['total_cost']:.2f}")
    print(f"-> 越限次数: {result['violation_count']} 次 ")


# ==========================================
# 主流程
# ==========================================
if __name__ == "__main__":
    # 管存容积敏感性测试: 扫描不同 C 值
    C_SENSITIVITY_VALUES = [
        ([0.02, 0.02], "小容积 (0.02)"),
        ([0.05, 0.05], "基准容积 (0.05)"),
        ([0.10, 0.10], "大容积 (0.10)"),
        ([0.15, 0.15], "超大容积 (0.15)"),
    ]

    print("=" * 85)
    print("   管存容积敏感性测试: 规则调度 vs 遗传算法 vs 动态规划 vs 强化学习 (SAC)")
    print("=" * 85)

    # 1. 生成测试场景 (所有 C 值共用同一组场景)
    print(f"\n📋 生成 {N_TEST_SCENARIOS} 个随机测试场景...")
    scenarios = generate_test_scenarios(N_TEST_SCENARIOS)

    sensitivity_summary = []

    for c_val, c_label in C_SENSITIVITY_VALUES:
        print(f"\n{'=' * 85}")
        print(f"   测试管存容积 C = {c_val}  ({c_label})")
        print(f"{'=' * 85}")

        # 2. 训练 RL
        print(f"\n🧠 训练 SAC 策略 ({RL_TRAINING_STEPS} 步)...")
        train_env = TopologicalDynamicEnv(noise_scale=0.15, C=c_val)
        model = SAC("MlpPolicy", train_env, verbose=0, learning_rate=0.001)
        callback = TrainingProgressBarCallback(total_timesteps=RL_TRAINING_STEPS)
        start = time.time()
        model.learn(total_timesteps=RL_TRAINING_STEPS, callback=callback)
        rl_train_time = time.time() - start
        print(f"\n✅ 训练完成！耗时: {rl_train_time:.2f} 秒")

        # 3. 评估
        eval_env = TopologicalDynamicEnv(noise_scale=0.0, C=c_val)
        rl_policy = make_rl_policy(model)

        rule_costs, ga_costs, dp_costs, rl_costs = [], [], [], []
        rule_violations, ga_violations, dp_violations, rl_violations = [], [], [], []
        rule_powers, ga_powers, dp_powers, rl_powers = [], [], [], []
        rule_times, ga_times, dp_times, rl_times = [], [], [], []

        detail_results = {"rule": None, "ga": None, "dp": None, "rl": None}
        ga_rng = np.random.default_rng(42)

        print(f"\n📊 在 {N_TEST_SCENARIOS} 个随机场景上评估...\n")

        for idx, (demand, prices) in enumerate(scenarios):
            t0 = time.time()
            r_rule = run_episode(eval_env, rule_policy, demand, prices)
            rule_times.append(time.time() - t0)
            rule_costs.append(r_rule["total_cost"])
            rule_violations.append(r_rule["violation_count"])
            rule_powers.append(r_rule["total_power"])

            t0 = time.time()
            best_sched = ga_optimize_scenario(BASE_DEMAND, prices, rng=ga_rng, C=c_val)
            def _ga_policy(obs, env, hour, _s=best_sched):
                return np.array([_s[hour]], dtype=np.float32)
            r_ga = run_episode(eval_env, _ga_policy, demand, prices)
            ga_times.append(time.time() - t0)
            ga_costs.append(r_ga["total_cost"])
            ga_violations.append(r_ga["violation_count"])
            ga_powers.append(r_ga["total_power"])

            t0 = time.time()
            dp_sched = dp_optimize_scenario(BASE_DEMAND, prices, C=c_val)
            def _dp_policy(obs, env, hour, _s=dp_sched):
                return np.array([_s[hour]], dtype=np.float32)
            r_dp = run_episode(eval_env, _dp_policy, demand, prices)
            dp_times.append(time.time() - t0)
            dp_costs.append(r_dp["total_cost"])
            dp_violations.append(r_dp["violation_count"])
            dp_powers.append(r_dp["total_power"])

            t0 = time.time()
            r_rl = run_episode(eval_env, rl_policy, demand, prices)
            rl_times.append(time.time() - t0)
            rl_costs.append(r_rl["total_cost"])
            rl_violations.append(r_rl["violation_count"])
            rl_powers.append(r_rl["total_power"])

            if idx == 0:
                detail_results = {"rule": r_rule, "ga": r_ga, "dp": r_dp, "rl": r_rl}

            print(f"场景 {idx+1:02d} | 规则: ¥{r_rule['total_cost']:>8.2f} ({r_rule['violation_count']}违) | "
                  f"GA: ¥{r_ga['total_cost']:>8.2f} ({r_ga['violation_count']}违) | "
                  f"DP: ¥{r_dp['total_cost']:>8.2f} ({r_dp['violation_count']}违) | "
                  f"RL: ¥{r_rl['total_cost']:>8.2f} ({r_rl['violation_count']}违)")

        # 汇总当前 C 值结果
        print(f"\n--- C={c_val} 汇总 ---")
        print(f"{'方法':<20} | {'平均成本':<12} | {'平均能耗':<12} | {'平均越限次数':<10} | {'平均耗时':<10}")
        print("-" * 80)
        print(f"{'传统规则调度':<16} | ¥{np.mean(rule_costs):>9.2f} | {np.mean(rule_powers):>9.2f} kWh | {np.mean(rule_violations):>8.1f} | {np.mean(rule_times)*1000:>7.1f} ms")
        print(f"{'遗传算法(离线)':<14} | ¥{np.mean(ga_costs):>9.2f} | {np.mean(ga_powers):>9.2f} kWh | {np.mean(ga_violations):>8.1f} | {np.mean(ga_times):>7.2f} s")
        print(f"{'动态规划(离线)':<14} | ¥{np.mean(dp_costs):>9.2f} | {np.mean(dp_powers):>9.2f} kWh | {np.mean(dp_violations):>8.1f} | {np.mean(dp_times)*1000:>7.1f} ms")
        print(f"{'强化学习 SAC':<15} | ¥{np.mean(rl_costs):>9.2f} | {np.mean(rl_powers):>9.2f} kWh | {np.mean(rl_violations):>8.1f} | {np.mean(rl_times)*1000:>7.1f} ms")
        print("-" * 80)

        sensitivity_summary.append({
            "c_label": c_label,
            "c_val": c_val,
            "rule": {"cost": np.mean(rule_costs), "power": np.mean(rule_powers), "viol": np.mean(rule_violations)},
            "ga":   {"cost": np.mean(ga_costs),   "power": np.mean(ga_powers),   "viol": np.mean(ga_violations)},
            "dp":   {"cost": np.mean(dp_costs),   "power": np.mean(dp_powers),   "viol": np.mean(dp_violations)},
            "rl":   {"cost": np.mean(rl_costs),   "power": np.mean(rl_powers),   "viol": np.mean(rl_violations)},
        })

    # ==========================================
    # 敏感性汇总表
    # ==========================================
    print("\n" + "=" * 100)
    print("   管存容积敏感性测试汇总 (平均成本 / 平均越限)")
    print("=" * 100)
    print(f"\n{'容积系数':<20} | {'规则调度':<18} | {'遗传算法':<18} | {'动态规划':<18} | {'强化学习':<18}")
    print("-" * 100)
    for s in sensitivity_summary:
        print(
            f"{s['c_label']:<16} | "
            f"¥{s['rule']['cost']:>7.2f} ({s['rule']['viol']:.1f}违) | "
            f"¥{s['ga']['cost']:>7.2f} ({s['ga']['viol']:.1f}违) | "
            f"¥{s['dp']['cost']:>7.2f} ({s['dp']['viol']:.1f}违) | "
            f"¥{s['rl']['cost']:>7.2f} ({s['rl']['viol']:.1f}违)"
        )
    print("-" * 100)

    # 各方法随 C 变化的成本趋势
    print(f"\n管存容积对各方法成本的影响:")
    for method, label in [("rule", "规则调度"), ("ga", "遗传算法"), ("dp", "动态规划"), ("rl", "强化学习")]:
        costs = [s[method]["cost"] for s in sensitivity_summary]
        labels = [s["c_label"] for s in sensitivity_summary]
        trend = "  |  ".join(f"{l}: ¥{c:.2f}" for l, c in zip(labels, costs))
        print(f"  {label}: {trend}")

    print(f"\n信息结构: 所有方法均知晓完整24h电价序列，但只能看到当前时刻的实际需求")
    print(f"  GA/DP 使用历史基准需求做离线规划，执行时面对实际(含噪声)需求")
    print(f"  RL/规则 基于当前观测做在线决策，RL 观测中包含完整电价序列")