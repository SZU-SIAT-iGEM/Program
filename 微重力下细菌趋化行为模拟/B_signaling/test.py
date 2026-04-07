import unittest
import numpy as np

# 假设你的模块名为 signaling，这里我们将你之前写的类直接放在这里以供测试，
# 实际项目中你只需要: from signaling import SignalingPathway
class SignalingPathway:
    def __init__(self, config, chemotaxis_mode: str):
        valid_modes = ["wild_type", "no_chemotaxis", "enhanced"]
        if chemotaxis_mode not in valid_modes:
            raise ValueError(f"无效的 chemotaxis_mode: {chemotaxis_mode}。必须为 {valid_modes} 之一。")
            
        self.mode = chemotaxis_mode
        self.tau_a = 0.1  
        self.tau_m = config.ADAPTATION_TIME  
        self.a0 = config.CW_BIAS_BASELINE  
        self.H = config.HILL_COEFF  
        self.a_half = 0.5  
        
        if self.mode == "enhanced":
            self.K_D = config.K_D / 5.0
        else:
            self.K_D = config.K_D

    def update(self, bacterium, current_concentration: float, dt: float):
        if self.mode == "no_chemotaxis":
            bacterium.set_tumble_probability(self.a0)
            return

        a = bacterium.receptor_activity
        m = bacterium.methylation
        c = current_concentration

        f_mc = m / (m + (1.0 + c / self.K_D) * (1.0 - m))
        da_dt = (f_mc - a) / self.tau_a
        dm_dt = (self.a0 - a) / self.tau_m

        a_new = a + da_dt * dt
        m_new = m + dm_dt * dt

        a_new = np.clip(a_new, 0.01, 0.99)
        m_new = np.clip(m_new, 0.01, 0.99)

        bacterium.receptor_activity = a_new
        bacterium.methylation = m_new

        p_tumble = (a_new ** self.H) / (a_new ** self.H + self.a_half ** self.H)
        bacterium.set_tumble_probability(p_tumble)


# ==================== 测试前置准备 ====================

class DummyConfig:
    """模拟 config.py 里的参数"""
    ADAPTATION_TIME = 4.0
    CW_BIAS_BASELINE = 0.35
    HILL_COEFF = 10.3
    K_D = 2.5e-6
    DT = 0.1

class DummyBacterium:
    """模拟细菌对象，用于测试状态读写"""
    def __init__(self):
        # 初始状态随意设定，测试会验证它们是否能收敛
        self.receptor_activity = 0.5
        self.methylation = 0.5
        self.tumble_probability = 0.5

    def set_tumble_probability(self, p):
        self.tumble_probability = np.clip(p, 0.0, 1.0)


# ==================== 单元测试类 ====================

class TestSignalingPathway(unittest.TestCase):
    def setUp(self):
        """每个测试用例运行前的初始化"""
        self.config = DummyConfig()
        self.dt = self.config.DT
        self.a0 = self.config.CW_BIAS_BASELINE
        
        # 预先计算基线概率 (稳态时的翻转概率)
        self.baseline_p_tumble = (self.a0 ** self.config.HILL_COEFF) / \
                                 (self.a0 ** self.config.HILL_COEFF + 0.5 ** self.config.HILL_COEFF)

    def test_no_chemotaxis_mode(self):
        """测试 1: 'no_chemotaxis' 模式下，翻转概率恒定不变"""
        pathway = SignalingPathway(self.config, "no_chemotaxis")
        bacterium = DummyBacterium()
        
        # 测试不同浓度和多次迭代
        test_concentrations = [0.0, 1e-6, 1e-3, 1.0]
        for conc in test_concentrations:
            pathway.update(bacterium, conc, self.dt)
            self.assertEqual(bacterium.tumble_probability, self.a0, 
                             f"浓度为 {conc} 时，无趋化模式的翻转概率偏离了基线。")

    def test_exact_adaptation_steady_state(self):
        """测试 2: 稳态测试（精确适应）。恒定浓度运行后，受体活性回到 a0 附近"""
        pathway = SignalingPathway(self.config, "wild_type")
        bacterium = DummyBacterium()
        
        constant_conc = 1e-5  # 恒定环境浓度 10 μM
        
        # 运行 1000 步 (100秒，远大于适应时间常数 4秒)
        for _ in range(1000):
            pathway.update(bacterium, constant_conc, self.dt)
            
        # 误差允许范围：±5% 的 a0
        tolerance = self.a0 * 0.05
        self.assertAlmostEqual(bacterium.receptor_activity, self.a0, delta=tolerance,
                               msg="系统未能在恒定浓度下精确适应回受体活性基线。")

    def test_step_response_and_recovery(self):
        """测试 3: 阶跃响应测试。浓度突升导致 run 延长（概率下降），随后慢慢恢复"""
        pathway = SignalingPathway(self.config, "wild_type")
        bacterium = DummyBacterium()
        
        # 第一阶段：在无引诱物(浓度为0)环境下达到稳态
        for _ in range(200):
            pathway.update(bacterium, 0.0, self.dt)
            
        initial_p_tumble = bacterium.tumble_probability
        
        # 验证初始已经处于稳态
        self.assertAlmostEqual(initial_p_tumble, self.baseline_p_tumble, delta=0.01)

        # 第二阶段：浓度阶跃，突然升高到 10 μM
        step_conc = 1e-5
        
        # 仅更新 1-2 步，观察快速响应
        pathway.update(bacterium, step_conc, self.dt)
        pathway.update(bacterium, step_conc, self.dt)
        
        drop_p_tumble = bacterium.tumble_probability
        # 断言翻转概率显著下降（Run行为延长）
        self.assertLess(drop_p_tumble, initial_p_tumble * 0.5, 
                        "浓度突升后，翻转概率没有立即显著下降。")
        
        # 第三阶段：适应恢复 (运行约 8 秒，2个适应时间常数)
        # 4秒时恢复大约 63%，8秒时恢复超过 86%
        for _ in range(80): 
            pathway.update(bacterium, step_conc, self.dt)
            
        recovered_p_tumble = bacterium.tumble_probability
        # 断言已经逐渐恢复并接近基线
        self.assertGreater(recovered_p_tumble, drop_p_tumble, "适应机制失效，翻转概率没有回升。")
        self.assertAlmostEqual(recovered_p_tumble, self.baseline_p_tumble, delta=0.05,
                               msg="经过充足时间后，翻转概率没有恢复到基线。")

    def test_enhanced_mode_transient_amplitude(self):
        """测试 4: 'enhanced' 模式比 'wild_type' 对同样刺激的瞬态响应幅度更大"""
        bacterium_wt = DummyBacterium()
        bacterium_enh = DummyBacterium()
        
        pathway_wt = SignalingPathway(self.config, "wild_type")
        pathway_enh = SignalingPathway(self.config, "enhanced")
        
        # 共同达到初始稳态 (浓度=0)
        for _ in range(200):
            pathway_wt.update(bacterium_wt, 0.0, self.dt)
            pathway_enh.update(bacterium_enh, 0.0, self.dt)
            
        # 施加微小的阶跃刺激 (1 μM，方便比较灵敏度)
        small_step_conc = 1e-6
        
        # 更新1步，记录瞬态下降
        pathway_wt.update(bacterium_wt, small_step_conc, self.dt)
        pathway_enh.update(bacterium_enh, small_step_conc, self.dt)
        
        # 因为 enhanced 更敏感 (K_D 更小)，在相同的正向浓度阶跃下，
        # a(t) 下降更深，导致翻转概率 P_tumble 下降得更多(即值更小)
        p_wt = bacterium_wt.tumble_probability
        p_enh = bacterium_enh.tumble_probability
        
        self.assertLess(p_enh, p_wt, 
                        "增强模式(enhanced)下的响应幅度没有大于野生型(wild_type)。")


if __name__ == '__main__':
    unittest.main(verbosity=2)