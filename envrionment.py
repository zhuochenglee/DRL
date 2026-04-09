import numpy as np
from scipy.optimize import root

# ==========================================
# 第一部分：管网拓扑与参数初始化
# ==========================================

# 1. 节点外部供需 Qs (单位模拟值)
# 规则：正值为气源注入，负值为用户消耗
Qs = np.array([400.0, -150.0, -250.0])  # 节点 0 供气400，节点 1 消耗150，节点 2 消耗250

# 2. 管道综合阻力系数 K (论文公式 2 的化简表示)
K = np.array([0.02, 0.05])  # 管道 1 (0->1) 和 管道 2 (1->2) 的阻力系数

# 3. 关联矩阵 A (Incidence Matrix)
# 规则：流出节点为 -1，流入节点为 1
# 形状：3行(节点) x 2列(管道)
A = np.array([
    [-1,  0],  
    [ 1, -1],  
    [ 0,  1]   
])

# 已知边界条件：参考节点的压力
p0_ref = 110.0  # 节点 0 的压力 (bar)

# ==========================================
# 第二部分：构建系统方程组 (残差函数)
# 对应论文中的核心方程组: AQ = Qs 和 A^T*p^2 = K*Q*|Q|
# ==========================================
def pipeline_equations(vars):
    # 解析未知变量: vars = [p1, p2, Q1, Q2]
    p = np.array([p0_ref, vars[0], vars[1]])  # 全网节点压力向量
    Q = np.array([vars[2], vars[3]])          # 全网管道流量向量
    
    # 获取压力的平方
    p_sq = p**2
    
    # --- 方程 1: 节点质量守恒 (论文公式 12 & 16) ---
    # 计算内部管网的净流动: A * Q
    internal_flow = np.dot(A, Q)
    # 质量平衡残差: 内部净流动必须等于外部供需 Qs
    mass_balance_error = internal_flow + Qs
    
    # --- 方程 2: 管道阻力压降 (论文公式 1 & 16) ---
    # 利用关联矩阵转置自动计算压差: (下游压力^2 - 上游压力^2)
    # 为了符合 p_up^2 - p_down^2，这里加个负号: - (A.T * p^2)
    pressure_drop = -np.dot(A.T, p_sq)
    # 计算阻力做功: K * Q * |Q|
    friction_loss = K * Q * np.abs(Q)
    # 阻力残差: 压差必须等于摩擦阻力
    pressure_error = pressure_drop - friction_loss
    
    # 组装返回所有残差 (对于含有参考压力的系统，省略参考节点的质量平衡方程以避免线性相关)
    # eq1: 节点 1 的质量守恒残差
    # eq2: 节点 2 的质量守恒残差
    # eq3: 管道 1 的压降残差
    # eq4: 管道 2 的压降残差
    return [mass_balance_error[1], mass_balance_error[2], pressure_error[0], pressure_error[1]]

# ==========================================
# 第三部分：调用 Levenberg-Marquardt (L-M) 求解器
# ==========================================

# 1. 提供一个初始猜测值 (Initial Guess) -> [p1, p2, Q1, Q2]
# 在非线性求解中，好的猜测值能加速收敛
initial_guess = [100.0, 90.0, 200.0, 200.0]

# 2. 调用 scipy 的 root 函数，明确指定 method='lm' (莱文贝格-马夸特方法)
print("开始 L-M 求解迭代...")
solution = root(pipeline_equations, initial_guess, method='lm')

# ==========================================
# 第四部分：解析输出结果
# ==========================================
if solution.success:
    print("\n 求解完成")
    print("-" * 30)
    print(f"节点 1 压力 (p1) : {solution.x[0]:.2f} bar")
    print(f"节点 2 压力 (p2) : {solution.x[1]:.2f} bar")
    print(f"管道 1 流量 (Q1) : {solution.x[2]:.2f} m³/h")
    print(f"管道 2 流量 (Q2) : {solution.x[3]:.2f} m³/h")
    print("-" * 30)
    print(f"L-M 算法迭代次数 : {solution.nfev} 次")
else:
    print("\n 求解失败：", solution.message)