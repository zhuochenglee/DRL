import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.optimize import root
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import time

class SimpleCompressorEnv(gym.Env):
    """
    单压缩机管网的强化学习环境
    拓扑: 节点0 (气源) -> [压缩机] -> 节点1 -> [管道] -> 节点2 (用户)
    """
    def __init__(self):
        super(SimpleCompressorEnv, self).__init__()
        
        # 1. 定义动作空间 Action Space: 压缩机排气比 alpha (连续值，范围 1.0 到 2.0)
        self.action_space = spaces.Box(low=1.0, high=2.0, shape=(1,), dtype=np.float32)
        
        # 2. 定义状态空间 Observation Space: 用户需求量 (归一化) 和 电价状态 (0或1)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 物理常量
        self.p0_ref = 100.0  # 气源初始压力 (bar)
        self.K_pipe = 0.05   # 管道阻力系数
        self.p_min_safe = 40.0 # 节点2的安全最低交气压力 (bar)
        
        self.current_demand = 0.0

    def reset(self, seed=None):
        """每回合开始，随机生成一个新的用气需求"""
        super().reset(seed=seed)
        # 随机生成用户需求: 100 到 300 m3/h 之间
        self.current_demand = self.np_random.uniform(100.0, 300.0)
        
       # 随机生成一个电价状态: 0代表谷电(便宜, 0.3元), 1代表峰电(贵, 1.2元)
        self.price_state = np.random.choice([0.0, 1.0]) 
        self.current_price = 0.3 if self.price_state == 0.0 else 1.2
    
        # 归一化需求量
        normalized_demand = (self.current_demand - 200.0) / 100.0
    
        # 返回组装好的状态数组
        return np.array([normalized_demand, self.price_state], dtype=np.float32), {}

    def step(self, action):
        """Agent 下达控制指令后，环境的物理反馈"""
        alpha = action[0]
        
        # --- 物理引擎开始 ---
        # 1. 压缩机做功，提升节点 1 的压力
        p1 = self.p0_ref * alpha
        
        # 2. 调用管道物理方程求解节点 2 的真实压力
        # 方程: p1^2 - p2^2 = K * Q^2
        def pipe_equation(vars):
            p2 = vars[0]
            # 这里流量已知等于需求量
            return p1**2 - p2**2 - self.K_pipe * (self.current_demand**2)
            
        sol = root(pipe_equation, x0=[p1 - 10]) # 给个初始猜测值
        p2_actual = sol.x[0] if sol.success else 0.0
        
        # 3. 计算压缩机能耗 (这里用简化公式: 流量 * (压比 - 1) 代替复杂的经验多项式)
        power_consumption = self.current_demand * (alpha - 1.0)
        
        # --- RDPO 奖励导向策略优化 ---
        reward = -(power_consumption * self.current_price) # 目标是最小化电费
        irr = 0 # 违规次数
        
        # 检查是否触碰物理红线 (例如：输送到用户的压力低于 40 bar)
        if p2_actual < self.p_min_safe or not sol.success:
            irr += 1
            reward -= 1e6 * irr  # 触发 10^6 的天价惩罚！
            
        # 步数结算 (因为是稳态单步 MDP，执行一次即结束当前回合)
        terminated = True
        truncated = False
        
        # 构造下一个状态 (在这里不需要，因为回合已结束)
        next_state = np.array([(self.current_demand - 200.0) / 100.0, self.price_state], dtype=np.float32)
        
        info = {
            "p1": p1,
            "p2": p2_actual,
            "power": power_consumption,
            "violation": irr > 0
        }
        
        return next_state, float(reward), terminated, truncated, info
    
if __name__ == "__main__":
    print("🚀 正在初始化管网 RL 环境...")
    env = SimpleCompressorEnv()
    
    check_env(env)
    print("环境检查通过")
    
    # 2. 实例化 SAC 算法
    # MlpPolicy 表示使用全连接多层感知机作为 Actor 和 Critic 的神经网络
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.001)
    
    # 3. 训练开始：让 Agent 在游乐场里试错 5000 次
    print("\n 开始训练 Agent 将进行 5000 次学习")
    start_time = time.time()
    model.learn(total_timesteps=5000)
    print(f" 训练完成！耗时: {time.time() - start_time:.2f} 秒")
    
   
    print("\n=== 测试训练好的 Agent ===")
    obs, info = env.reset()
    print(f"用户需求量: {env.current_demand:.2f} m³/h")
    
    # 注意：测试时 deterministic=True，表示关闭随机探索，直接输出网络认为最优的动作
    action, _states = model.predict(obs, deterministic=True)
    
    # 将 AI 计算出的动作输入物理环境
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n AI 决定将排气比设定为: {action[0]:.4f}")
    print(f"压缩机出口压力 (p1): {info['p1']:.2f} bar")
    print(f"用户端接收压力 (p2): {info['p2']:.2f} bar")
    print(f"压缩机总能耗: {info['power']:.2f} kW")
    
    if info['violation']:
        print(f"安全状态: 违反安全红线 Agent 被扣了 {reward:.0f} 分。")
    else:
        print(f"安全状态: 安全运行。")
        print(f"💰 最终得分: {reward:.2f} ")