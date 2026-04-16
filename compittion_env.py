import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import time

class LifecycleYBranchEnv(gym.Env):
    """
    全生命周期 (365天) Y型管网多机组协同 RL 环境
    """
    def __init__(self):
        super(LifecycleYBranchEnv, self).__init__()
        
        # ==========================================
        # 1. 动作与状态空间
        # ==========================================
        # 动作 (4维): [总体排气比, 机组1意愿, 机组2意愿, 机组3意愿]
        # 值域 [-1, 1]，方便神经网络输出
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 状态 (10维): [归一化步数, K1需求, K2需求, 电价, 目标红线压力, 集输管存, K1管存, K2管存, 上次启停动作...]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32)
        
        # ==========================================
        # 2. 物理与拓扑参数
        # ==========================================
        # Y型关联矩阵: 5节点 x 3管道
        self.A = np.array([
            [-1,  0,  0],  # 0: 气源
            [ 1,  0,  0],  # 1: 集输站
            [ 0, -1, -1],  # 2: 压缩机出口分流点
            [ 0,  1,  0],  # 3: 井口 K1
            [ 0,  0,  1]   # 4: 井口 K2
        ])
        
        self.K = np.array([0.02, 0.05, 0.08]) # 管道阻力 (K1支线容易，K2支线难)
        self.C = np.array([0.05, 0.02, 0.02]) # 管存容积系数
        self.p0_ref = 60.0 # 气源压力 (6 MPa)
        
        # 机组铭牌功率 (kW)
        self.COMP_CAPACITY = np.array([4500.0, 4500.0, 4000.0])
        
        # ==========================================
        # 3. 周期工况定义 (365天 = 8760小时)
        # ==========================================
        self.max_hours = 365 * 24
        self.current_hour = 0
        
        # 初始化动态状态
        self.P_internal = np.array([60.0, 100.0, 100.0]) # [集输P1, 井口P3, 井口P4] (单位: bar)
        self.last_u = np.array([1.0, 1.0, 1.0]) # 初始默认三台全开

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_hour = 0
        self.P_internal = np.array([60.0, 100.0, 100.0])
        self.last_u = np.array([1.0, 1.0, 1.0])
        return self._get_obs(), {}

    def _get_obs(self):
        # 计算当前生命周期进度 (0.0 到 1.0)
        progress = self.current_hour / self.max_hours
        
        # 1. 计算流量衰减: 600万方/天 -> 270万方/天 (换算为小时: 25 -> 11.25)
        total_demand = 25.0 - (25.0 - 11.25) * progress
        # 假设 K1 占 60%，K2 占 40%
        demand_k1 = total_demand * 0.6
        demand_k2 = total_demand * 0.4
        
        # 2. 计算压力上升: 10 MPa (100 bar) -> 29 MPa (290 bar)
        p_min_safe = 100.0 + (290.0 - 100.0) * progress
        
        # 3. 模拟电价 (假定 8-22点为峰电)
        hour_of_day = self.current_hour % 24
        price = 1.2 if 8 <= hour_of_day <= 22 else 0.3
        
        # 组装并归一化观测向量 (-1 到 1 之间，利于神经网络学习)
        obs = np.array([
            progress * 2.0 - 1.0,
            (demand_k1 - 10.0) / 10.0,
            (demand_k2 - 10.0) / 10.0,
            1.0 if price > 0.5 else -1.0,
            (p_min_safe - 195.0) / 95.0,
            (self.P_internal[0] - 180.0) / 170.0,
            (self.P_internal[1] - 180.0) / 170.0,
            (self.P_internal[2] - 180.0) / 170.0,
            self.last_u[0], self.last_u[1], self.last_u[2]
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        progress = self.current_hour / self.max_hours
        total_demand = 25.0 - (25.0 - 11.25) * progress
        demand_k1 = total_demand * 0.6
        demand_k2 = total_demand * 0.4
        p_min_safe = 100.0 + (290.0 - 100.0) * progress
        hour_of_day = self.current_hour % 24
        price = 1.2 if 8 <= hour_of_day <= 22 else 0.3
        
        # ==========================================
        # 1. 动作解码 (混合整数)
        # ==========================================
        # 后期压力高达 290 bar，排气比最高可能需要 6.0
        alpha = 3.5 + action[0] * 2.5 
        
        # 截断法控制启停
        u = np.array([
            1.0 if action[1] > 0 else 0.0,
            1.0 if action[2] > 0 else 0.0,
            1.0 if action[3] > 0 else 0.0
        ])
        if np.sum(u) == 0: u[2] = 1.0 # 强制保底开最小的一台
            
        # ==========================================
        # 2. 拓扑矩阵计算物理流场
        # ==========================================
        P_nodes = np.array([
            self.p0_ref,                 # 0: 源
            self.P_internal[0],          # 1: 集输
            self.P_internal[0] * alpha,  # 2: 压后
            self.P_internal[1],          # 3: K1
            self.P_internal[2]           # 4: K2
        ])
        
        dP = -np.dot(self.A.T, P_nodes**2)
        Q = np.sign(dP) * np.sqrt(np.abs(dP) / self.K)
        
        Q_comp = Q[1] + Q[2] 
        net_flows = np.array([
            Q[0] - Q_comp,     # 集输收支
            Q[1] - demand_k1,  # K1收支
            Q[2] - demand_k2   # K2收支
        ])
        
        self.P_internal += self.C * net_flows
        self.P_internal = np.clip(self.P_internal, 10.0, 350.0) # 放宽物理上限以容纳 290 bar
        
        # ==========================================
        # 3. 经济账与奖惩计算
        # ==========================================
        # 3.1 理论做功 
        theoretical_power = max(0, Q_comp) * (alpha - 1.0) * 150.0 
        
        # 3.2 效率陷阱与物理超载限制
        active_capacity = np.sum(u * self.COMP_CAPACITY)
        
        # 防止分母为0
        if active_capacity == 0:
            active_capacity = 1e-5
            
        load_rate = theoretical_power / active_capacity 
        
        efficiency = 0.8 
        overload_penalty = 0.0
        
        # 极其严厉的物理特性曲线限制
        if load_rate < 0.4:
            efficiency = 0.4 # 低负荷低效
        if load_rate > 1.2:
            efficiency = 0.1 
            # 补丁1：机器严重超载（比如用4000kW干30000kW的活），直接给设备损坏天价惩罚！
            overload_penalty = 100000.0 
            
        # 补丁2：加入机组“怠速空转”耗电 (只要开着，哪怕不干活也要消耗 15% 额定功率)
        idle_power = active_capacity * 0.15
        actual_power = (theoretical_power / efficiency) + idle_power
        
        # 3.3 启停磨损惩罚 
        switch_count = np.sum(np.abs(u - self.last_u))
        switching_cost = switch_count * 1000.0 
        
        # 最终奖励 = -真实电费 - 磨损费 - 超载罚款
        reward = -(actual_power * price + switching_cost + overload_penalty) * 0.001 
        
        # 安全红线惩罚
        penalty_factor = 5000.0
        irr = 0
        for p in self.P_internal[1:]:
            if p < p_min_safe:
                irr += 1
                reward -= penalty_factor * (p_min_safe - p) * 0.001
                
        # 将 -5000.0 改为 -1000000.0 (一百万)
        reward = np.clip(reward, -1000000.0, 10.0)
                
        
        # 状态更新
        self.last_u = u.copy()
        self.current_hour += 1
        terminated = bool(self.current_hour >= self.max_hours)
        
        info = {
            "progress_day": self.current_hour // 24,
            "p_min_safe": p_min_safe,
            "p_K1": self.P_internal[1],
            "p_K2": self.P_internal[2],
            "active_units": np.sum(u),
            "actual_power": actual_power,
            "violation": irr > 0
        }
        
        return self._get_obs(), float(reward), terminated, False, info

# ==========================================
# 训练与测试代码
# ==========================================
if __name__ == "__main__":
    print("🚀 初始化 365 天注气生命周期 RL 环境...")
    env = LifecycleYBranchEnv()
    check_env(env)
    
    print("\n🧠 开始高强度训练 (约需要一两分钟)...")
    # 因为周期长达 8760 步，需要多训练几个 episode，这里设定 50,000 步演示
    model = SAC("MlpPolicy", env, verbose=0, learning_rate=0.001, batch_size=256)
    
    start_time = time.time()
    model.learn(total_timesteps=50000)
    print(f"✅ 训练完成！耗时: {time.time() - start_time:.2f} 秒")

    print("\n📊 抽取关键生命周期节点进行汇报:\n")
    print(f"{'天数':<5} | {'要求压力':<8} | {'K1到达压':<8} | {'K2到达压':<8} | {'启停机组':<18} | {'总能耗(kW)':<10}")
    print("-" * 75)
    
    obs, _ = env.reset()
    total_cost = 0
    
    for i in range(365 * 24):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 每天中午 12:00 打印一次日志，观察全年趋势
        if i % 24 == 12:
            u_state = env.last_u
            machine_str = f"[{'4500' if u_state[0] else '停'} | {'4500' if u_state[1] else '停'} | {'4000' if u_state[2] else '停'}]"
            print(f"第{info['progress_day']:03d}天 | {info['p_min_safe']:>5.1f} bar | {info['p_K1']:>6.1f} bar | {info['p_K2']:>6.1f} bar | {machine_str:<18} | {info['actual_power']:.1f}")