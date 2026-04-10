import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
import time

class TopologicalDynamicEnv(gym.Env):

    def __init__(self):
        super(TopologicalDynamicEnv, self).__init__()
        
        # 动作空间: 压缩机排气比 alpha，动作在节点 1 上面
        self.action_space = spaces.Box(low=1.0, high=2.0, shape=(1,), dtype=np.float32)
        
        # 状态空间: [用户需求, 电价状态, 节点2管存, 节点3管存] 
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
      
        # 关联矩阵 A: 3*2  节点*管道
        self.A = np.array([
            [-1,  0],  # 节点 1 (气源，含有压缩机)
            [ 1, -1],  # 节点 2 (中间站)
            [ 0,  1]   # 节点 3 (用户端)
        ])
        
        # 将矩阵切片：已知压力的源节点 vs 未知压力的内部节点
        self.A_source = self.A[0:1, :]    # shape (1, 2)
        self.A_internal = self.A[1:3, :]  # shape (2, 2)
        
        self.K = np.array([0.02, 0.05]) # 管道 1, 2 的阻力系数
        self.C = np.array([0.05, 0.05]) # 节点 2, 3 的管存容积弹性系数
        self.p0_ref = 100.0
        self.p_min_safe = 80.0
        self.p_max_safe = 150.0
        
        # 初始管存状态 (节点 2 和 节点 3 的初始压力)
        self.P_internal = np.array([100.0, 100.0]) 
        self.current_hour = 0
        
        # 外部环境时间序列
        self.demand_series = np.array([
            120, 110, 110, 115, 120, 130, 150, 180,
            220, 250, 260, 250, 240, 240, 250, 260,
            270, 280, 280, 260, 220, 180, 150, 130
        ])
        self.actual_prices = np.array([
            0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
            1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
            1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.3, 0.3
        ])

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_hour = 0
        # 重置所有内部节点的管存压力
        self.P_internal = np.array([100.0, 100.0]) 
        return self._get_obs(), {}
    
    def _get_obs(self):
        demand = self.demand_series[self.current_hour % 24]
        price_state = 1.0 if self.actual_prices[self.current_hour % 24] > 0.5 else 0.0
        
        norm_demand = (demand - 200.0) / 100.0
        norm_p2 = (self.P_internal[0] - 105.0) / 95.0
        norm_p3 = (self.P_internal[1] - 105.0) / 95.0
        
        return np.array([norm_demand, price_state, norm_p2, norm_p3], dtype=np.float32)

    def step(self, action):
        alpha = action[0]
        demand = self.demand_series[self.current_hour]
        price = self.actual_prices[self.current_hour]
       
        # 1. 设置气源节点压力向量 (压缩机做功后)
        P_source = np.array([self.p0_ref * alpha])
        
        # 2. 计算所有管道的压差: dP = -(A_source^T * P_source^2 + A_internal^T * P_internal^2)
        P_source_sq = P_source**2
        P_internal_sq = self.P_internal**2
        dP = -(np.dot(self.A_source.T, P_source_sq) + np.dot(self.A_internal.T, P_internal_sq))
        
        # 3. 计算所有管道的真实流量 Q 
        Q = np.sign(dP) * np.sqrt(np.abs(dP) / self.K)
        
        # 4. 计算内部节点的流量收支
        # Q_ext_internal: 外部对内部节点的净注入量。节点2为0，节点3为 -demand (抽出)
        Q_ext_internal = np.array([0.0, -demand])
        
        # 计算内部节点的净流入并更新管存压力: P_new = P_old + C * (A_internal * Q + Q_ext)
        net_flow_internal = np.dot(self.A_internal, Q) + Q_ext_internal
        self.P_internal += self.C * net_flow_internal
        
        # 物理限制
        self.P_internal = np.clip(self.P_internal, 10.0, 200.0)
        
        # 5. 计算奖励 (能耗与电价)
        # 气源输出的真实流量 Q_source = A_source * Q 
        # (因为 A_source 第一行是 -1, 所以流出量是负的，取绝对值)
        Q_source_actual = np.abs(np.dot(self.A_source, Q)[0])
        power_consumption = Q_source_actual * (alpha - 1.0)
        
        reward = -(power_consumption * price)
        irr = 0 
        
        # 检查所有内部节点的安全红线
        for p in self.P_internal:
            if p < self.p_min_safe:
                irr += 1
                reward -= 1000 * (self.p_min_safe - p)
            if p > self.p_max_safe:
                irr += 1
                reward -= 1000 * (p - self.p_max_safe)
                
        self.current_hour += 1
        terminated = bool(self.current_hour >= 24)
        
        info = {
            "p1_source": P_source[0],
            "p2_internal": self.P_internal[0],
            "p3_user": self.P_internal[1],
            "Q1": Q[0],
            "Q2": Q[1],
            "power": power_consumption,
            "violation": irr > 0
        }
        
        return self._get_obs(), float(reward), terminated, False, info


class TrainingProgressBarCallback(BaseCallback):

    def __init__(self, total_timesteps, bar_width=30):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.bar_width = bar_width
        self.update_interval = max(1, total_timesteps // 100)
        self.last_displayed_step = -1
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        print("\n 训练进度:")
        self._render_progress(0)

    def _on_step(self):
        current_step = min(self.num_timesteps, self.total_timesteps)
        if (
            current_step >= self.total_timesteps
            or current_step - self.last_displayed_step >= self.update_interval
        ):
            self.last_displayed_step = current_step
            self._render_progress(current_step)
        return True

    def _on_training_end(self):
        self._render_progress(self.total_timesteps)
        print()

    def _render_progress(self, current_step):
        progress = min(current_step / self.total_timesteps, 1.0)
        filled = int(self.bar_width * progress)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        elapsed = 0.0 if self.start_time is None else time.time() - self.start_time
        print(
            f"\r[{bar}] {progress:6.2%} ({current_step}/{self.total_timesteps}) | {elapsed:6.1f}s",
            end="",
            flush=True,
        )
    


if __name__ == "__main__":
    print("🚀 初始化拓扑动态管网 RL 环境...")
    env = TopologicalDynamicEnv()
    
    # 检查自定义环境是否符合 Gym 规范
    check_env(env)
    
    
    # 实例化 SAC 算法 
    
    model = SAC("MlpPolicy", env, verbose=0, learning_rate=0.001)
    
    total_timesteps = 200000
    progress_callback = TrainingProgressBarCallback(total_timesteps=total_timesteps)

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=progress_callback)
    print(f" 训练完成！耗时: {time.time() - start_time:.2f} 秒")


    print("\n执行 24 小时动态调度测试 \n")
    print(f"{'时间':<6} | {'电价':<4} | {'需求':<4} | {'动作(排比)':<8} | {'节点2压':<7} | {'节点3压':<7} | {'总能耗(kW)':<8}")
    print("-" * 75)
    
    obs, _ = env.reset()
    total_power = 0
    total_cost = 0
    violation_count = 0
    
    for i in range(24):
        # 让训练好的模型输出决策
        action, _ = model.predict(obs, deterministic=True)
        # 物理引擎执行决策
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_price = env.actual_prices[i]
        current_demand = env.demand_series[i]
        
        total_power += info['power']
        total_cost += info['power'] * current_price
        if info['violation']:
            violation_count += 1
            
        print(f"{i:02d}:00  | {current_price:.1f}元 | {current_demand:<4} | {action[0]:<8.4f} | {info['p2_internal']:<7.1f} | {info['p3_user']:<7.1f} | {info['power']:<8.1f}")

    print("-" * 75)
    print(f"\n24 小时运行总结报告:")
    print(f"-> 全天总耗电量: {total_power:.2f} kWh")
    print(f"-> 全天总运行成本: ¥ {total_cost:.2f}")
    print(f"-> 越限次数: {violation_count} 次 (预期应为 0)")