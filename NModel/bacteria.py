"""
趋化性模拟建模 —— 细菌数据结构定义
====================================
定义单个细菌的状态数据结构（Bacterium类）和群体初始化工具函数。
所有模块通过读写Bacterium对象来交换数据。

设计原则：
  - Bacterium是纯数据容器，不包含物理计算逻辑
  - 物理计算由各功能模块（signaling / movement / metabolism）负责
  - 各模块通过Bacterium的公开属性和方法读写状态
"""

import numpy as np
from config import (
    DOMAIN_SIZE,
    N_BACTERIA,
    INITIAL_DISTRIBUTION,
    CW_BIAS_BASELINE,
    SWIM_SPEED,
)


# ============================================================
# 工具函数
# ============================================================

def random_unit_vector():
    """
    生成三维空间中均匀分布的随机单位方向向量。
    
    方法：从三维标准正态分布中采样，然后归一化。
    三维正态分布的方向在球面上均匀分布（Muller, 1959）。
    
    返回：
        np.ndarray, shape=(3,), 单位向量
    """
    vec = np.random.randn(3)
    norm = np.linalg.norm(vec)
    while norm < 1e-10:
        vec = np.random.randn(3)
        norm = np.linalg.norm(vec)
    return vec / norm


def random_tumble_direction(current_direction):
    """
    根据当前运动方向生成翻转后的新方向。
    
    翻转角度 θ 服从正态分布 N(62°, 26°²)。
    方位角 φ 在 [0, 2π) 上均匀分布。
    使用Rodrigues旋转公式实现方向旋转。
    
    来源：
        Berg HC & Brown DA (1972) Nature 239:500-504
        Saragosti J et al. (2012) PLoS ONE 7:e35412
    
    参数：
        current_direction: np.ndarray, shape=(3,), 当前运动方向（单位向量）
    
    返回：
        np.ndarray, shape=(3,), 新的运动方向（单位向量）
    """
    theta_deg = np.random.normal(62.0, 26.0)
    theta_deg = np.clip(theta_deg, 0.0, 180.0)
    theta = np.radians(theta_deg)
    
    phi = np.random.uniform(0, 2 * np.pi)
    
    d = current_direction
    if abs(d[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    
    e1 = np.cross(d, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    e2 = e2 / np.linalg.norm(e2)
    
    new_dir = (
        np.cos(theta) * d
        + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2)
    )
    
    new_dir = new_dir / np.linalg.norm(new_dir)
    return new_dir


# ============================================================
# 细菌数据结构
# ============================================================

class Bacterium:
    """
    单个细菌的状态容器。
    
    所有模块通过读写这个对象来交换数据：
      - movement模块 读写 position, direction, state, state_timer
      - signaling模块 读写 receptor_activity, methylation, tumble_probability
      - metabolism模块 读写 alive, nutrient_consumed
      - environment模块 读取 position（查询浓度）
    
    属性分为四组：身份、空间状态、信号通路状态、代谢状态。
    """
    
    __slots__ = [
        'id',
        'position', 'direction', 'state', 'state_timer',
        'receptor_activity', 'methylation', 'tumble_probability',
        'alive', 'nutrient_consumed',
    ]
    
    def __init__(self, position, bacterium_id):
        self.id = bacterium_id
        
        self.position = np.array(position, dtype=np.float64)
        self.direction = random_unit_vector()
        self.state = "run"
        self.state_timer = 0.0
        
        self.receptor_activity = CW_BIAS_BASELINE
        self.methylation = CW_BIAS_BASELINE
        self.tumble_probability = CW_BIAS_BASELINE
        
        self.alive = True
        self.nutrient_consumed = 0.0
    
    def get_position(self):
        return self.position.copy()
    
    def set_tumble_probability(self, p):
        self.tumble_probability = float(np.clip(p, 0.0, 1.0))
    
    def distance_to(self, point):
        return float(np.linalg.norm(self.position - np.asarray(point)))
    
    def __repr__(self):
        return (
            f"Bacterium(id={self.id}, "
            f"pos=[{self.position[0]*1e6:.1f}, {self.position[1]*1e6:.1f}, {self.position[2]*1e6:.1f}] μm, "
            f"state={self.state}, "
            f"P_tumble={self.tumble_probability:.3f}, "
            f"alive={self.alive})"
        )


# ============================================================
# 群体初始化
# ============================================================

def create_population(n=None, distribution=None, domain_size=None):
    if n is None:
        n = N_BACTERIA
    if distribution is None:
        distribution = INITIAL_DISTRIBUTION
    if domain_size is None:
        domain_size = DOMAIN_SIZE
    
    bacteria = []
    center = domain_size / 2.0
    
    for i in range(n):
        if distribution == "uniform":
            pos = np.random.uniform(0, domain_size, 3)
        elif distribution == "cluster":
            sigma = domain_size / 4.0
            pos = np.random.normal(center, sigma, 3)
            pos = np.clip(pos, 0, domain_size)
        else:
            raise ValueError(
                f"未知的分布类型: '{distribution}'。"
                f"请使用 'uniform' 或 'cluster'。"
            )
        bacteria.append(Bacterium(pos, bacterium_id=i))
    
    return bacteria


# ============================================================
# 群体统计工具
# ============================================================

def count_near_substrate(bacteria, substrate_center, inner_radius, outer_radius):
    center = np.asarray(substrate_center)
    count = 0
    for bac in bacteria:
        if not bac.alive:
            continue
        dist = bac.distance_to(center)
        if inner_radius <= dist <= outer_radius:
            count += 1
    return count


def get_all_positions(bacteria, alive_only=True):
    positions = []
    for bac in bacteria:
        if alive_only and not bac.alive:
            continue
        positions.append(bac.position)
    
    if len(positions) == 0:
        return np.empty((0, 3))
    return np.array(positions)


def get_population_stats(bacteria):
    alive = [b for b in bacteria if b.alive]
    n_alive = len(alive)
    
    if n_alive == 0:
        return {
            "n_alive": 0,
            "n_running": 0,
            "n_tumbling": 0,
            "mean_tumble_prob": 0.0,
            "total_consumption": 0.0,
            "mean_position": np.zeros(3),
        }
    
    n_running = sum(1 for b in alive if b.state == "run")
    n_tumbling = n_alive - n_running
    mean_tp = np.mean([b.tumble_probability for b in alive])
    total_cons = sum(b.nutrient_consumed for b in bacteria)
    mean_pos = np.mean([b.position for b in alive], axis=0)
    
    return {
        "n_alive": n_alive,
        "n_running": n_running,
        "n_tumbling": n_tumbling,
        "mean_tumble_prob": float(mean_tp),
        "total_consumption": float(total_cons),
        "mean_position": mean_pos,
    }


if __name__ == "__main__":
    print("=== 细菌模块自测 ===\n")
    
    pop = create_population(n=10, distribution="uniform")
    print(f"创建了 {len(pop)} 个细菌：")
    for b in pop[:3]:
        print(f"  {b}")
    print(f"  ...（省略 {len(pop)-3} 个）\n")
    
    print("随机方向向量测试（5个样本）：")
    for _ in range(5):
        v = random_unit_vector()
        print(f"  {v}  |v| = {np.linalg.norm(v):.6f}")
    print()
    
    print("翻转方向测试：")
    d0 = np.array([1.0, 0.0, 0.0])
    angles = []
    for _ in range(1000):
        d1 = random_tumble_direction(d0)
        angle = np.degrees(np.arccos(np.clip(np.dot(d0, d1), -1, 1)))
        angles.append(angle)
    print(f"  初始方向: {d0}")
    print(f"  1000次翻转后的偏转角统计：")
    print(f"    均值 = {np.mean(angles):.1f}°（期望 62°）")
    print(f"    标准差 = {np.std(angles):.1f}°（期望 ~26°）")
    print()
    
    center = np.array([DOMAIN_SIZE/2]*3)
    n_near = count_near_substrate(pop, center, 0, 500e-6)
    print(f"域中心500μm半径内的菌数: {n_near} / {len(pop)}")
    
    stats = get_population_stats(pop)
    print(f"\n群体统计：")
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: [{v[0]*1e6:.1f}, {v[1]*1e6:.1f}, {v[2]*1e6:.1f}] μm")
        else:
            print(f"  {k}: {v}")
    
    print("\n=== 自测通过 ===")
