import numpy as np

class SignalingPathway:
    """
    细菌趋化信号通路模块 (简化版 Barkai-Leibler 模型)
    负责计算受体活性、甲基化水平以及最终的翻转概率。
    """
    
    def __init__(self, config, chemotaxis_mode: str):
        """
        初始化信号通路模块。
        
        参数:
            config: 配置模块 (包含物理参数)
            chemotaxis_mode: "wild_type" / "no_chemotaxis" / "enhanced"
               - "no_chemotaxis": 翻转概率恒为 CW_BIAS_BASELINE
               - "wild_type": 正常执行 ODE
               - "enhanced": 灵敏度提高，K_D 降低5倍
        """
        valid_modes = ["wild_type", "no_chemotaxis", "enhanced"]
        if chemotaxis_mode not in valid_modes:
            raise ValueError(f"无效的 chemotaxis_mode: {chemotaxis_mode}。必须为 {valid_modes} 之一。")
            
        self.mode = chemotaxis_mode
        
        # === 导入并设置参数 ===
        self.tau_a = 0.1  # 受体活性的快速响应时间常数，单位 s
        self.tau_m = config.ADAPTATION_TIME  # 甲基化适应的慢时间常数，单位 s
        self.a0 = config.CW_BIAS_BASELINE  # 适应目标活性 (基线)
        self.H = config.HILL_COEFF  # Hill 系数
        self.a_half = 0.5  # 半最大活性参数
        
        # 根据趋化模式设置受体解离常数 K_D (单位 M)
        if self.mode == "enhanced":
            self.K_D = config.K_D / 5.0
        else:
            self.K_D = config.K_D

    def update(self, bacterium: 'Bacterium', current_concentration: float, dt: float):
        """
        更新一个细菌的信号通路状态和翻转概率。
        
        参数:
            bacterium: Bacterium 对象，表示当前正在更新的细菌
            current_concentration: 当前位置的引诱物浓度 (单位 M)
            dt: 时间步长 (单位 s)
        """
        # 1. "no_chemotaxis" 模式处理
        if self.mode == "no_chemotaxis":
            bacterium.set_tumble_probability(self.a0)
            return

        # 2. 读取当前状态
        a = bacterium.receptor_activity
        m = bacterium.methylation
        c = current_concentration

        # 3. 计算受体活性对甲基化和浓度的响应函数 f(m, c)
        # 来源: Barkai & Leibler (1997) Nature 387:913-917; Tu et al. (2008) PNAS 105:14855-14860.
        f_mc = m / (m + (1.0 + c / self.K_D) * (1.0 - m))

        # 4. 计算 ODE 的导数
        # da/dt = [f(m, c) - a] / tau_a
        # dm/dt = [a0 - a] / tau_m
        da_dt = (f_mc - a) / self.tau_a
        dm_dt = (self.a0 - a) / self.tau_m

        # 5. 欧拉法更新状态 (Euler method integration)
        a_new = a + da_dt * dt
        m_new = m + dm_dt * dt

        # 6. 将状态裁剪到 [0.01, 0.99] 范围内，防止数值不稳定或溢出
        a_new = np.clip(a_new, 0.01, 0.99)
        m_new = np.clip(m_new, 0.01, 0.99)

        # 回写更新后的状态到细菌对象
        bacterium.receptor_activity = a_new
        bacterium.methylation = m_new

        # 7. 将受体活性 a(t) 通过 Hill 函数映射为翻转概率
        # P_tumble = a^H / (a^H + a_1/2^H)
        # 来源: Cluzel et al. (2000) Science 287:1652-1655.
        p_tumble = (a_new ** self.H) / (a_new ** self.H + self.a_half ** self.H)

        # 更新细菌的翻转概率
        bacterium.set_tumble_probability(p_tumble)