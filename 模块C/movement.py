# movement.py
"""
细菌三维运动模块
实现 run-and-tumble 随机游走，包含重力沉降和边界反射。
所有物理参数从 config 对象获取，单位使用 SI。
"""

import numpy as np
from bacteria import Bacterium  # 假设 bacteria.py 在同一目录

class MovementEngine:
    """
    细菌运动引擎
    负责更新每个细菌的位置、方向和运动状态。
    """

    def __init__(self, config):
        """
        初始化运动引擎，从 config 提取所需参数。

        Parameters
        ----------
        config : module or SimpleNamespace
            包含全局参数的配置对象，必须包含以下属性：
            DOMAIN_SIZE, SWIM_SPEED, GRAVITY_ON, SEDIMENTATION_SPEED,
            MEAN_RUN_TIME, TUMBLE_DURATION, TUMBLE_ANGLE_MEAN, TUMBLE_ANGLE_STD
        """
        self.domain_size = config.DOMAIN_SIZE
        self.swim_speed = config.SWIM_SPEED                 # m/s
        self.gravity_on = config.GRAVITY_ON
        self.sedimentation_speed = config.SEDIMENTATION_SPEED  # m/s
        self.mean_run_time = config.MEAN_RUN_TIME           # s
        self.tumble_duration = config.TUMBLE_DURATION       # s
        # 翻转角度参数（度数）
        self.tumble_angle_mean = config.TUMBLE_ANGLE_MEAN    # degrees
        self.tumble_angle_std = config.TUMBLE_ANGLE_STD      # degrees

    def step(self, bacterium: Bacterium, dt: float):
        """
        执行一个细菌的一个时间步运动更新。

        根据细菌当前状态（run/tumble）更新位置、方向、状态计时器，
        并处理可能的翻转切换、重力沉降和边界反射。

        Parameters
        ----------
        bacterium : Bacterium
            待更新的细菌对象，其属性将被直接修改。
        dt : float
            时间步长，单位 s。
        """
        if bacterium.state == "run":
            # ----- run 状态：直线运动 -----
            # 位置更新：沿当前方向匀速运动
            # 来源：Berg & Brown (1972) Nature 239:500-504
            bacterium.position += self.swim_speed * bacterium.direction * dt

            # 重力沉降（如果开启）：z 方向额外向下位移
            # 来源：Cluzel et al. (2000) Science 287:1652-1655 中估算值
            if self.gravity_on:
                bacterium.position[2] -= self.sedimentation_speed * dt

            # 边界反射（确保位置在 [0, domain_size]^3 内）
            self._apply_boundary_reflection(bacterium)

            # 更新状态计时器
            bacterium.state_timer += dt

            # 判断是否发生翻转（随机过程）
            # 翻转概率：每单位时间概率 = tumble_probability / mean_run_time
            # 因此 dt 内触发概率 = tumble_probability * dt / mean_run_time
            # 来源：Berg & Brown (1972) 指数分布停留时间
            if np.random.random() < bacterium.tumble_probability * dt / self.mean_run_time:
                # 切换到 tumble 状态，重置计时器
                bacterium.state = "tumble"
                bacterium.state_timer = 0.0

        elif bacterium.state == "tumble":
            # ----- tumble 状态：原地旋转，位置不变 -----
            bacterium.state_timer += dt

            # 如果 tumble 持续时间结束，计算新方向并切回 run
            if bacterium.state_timer >= self.tumble_duration:
                # 计算新的运动方向
                new_direction = self._generate_new_direction(bacterium.direction)
                bacterium.direction = new_direction
                bacterium.state = "run"
                bacterium.state_timer = 0.0
        else:
            raise ValueError(f"未知的细菌状态: {bacterium.state}")

    def _apply_boundary_reflection(self, bacterium):
        """
        对单个细菌应用镜面反射边界条件。
        如果位置超出 [0, domain_size] 范围，将位置反射回域内，并反转对应速度分量方向。

        Parameters
        ----------
        bacterium : Bacterium
            待处理的细菌对象。
        """
        pos = bacterium.position
        direction = bacterium.direction
        L = self.domain_size

        # 对每个坐标轴分别处理
        # x 轴
        if pos[0] < 0:
            pos[0] = -pos[0]
            direction[0] = -direction[0]
        elif pos[0] > L:
            pos[0] = 2 * L - pos[0]
            direction[0] = -direction[0]

        # y 轴
        if pos[1] < 0:
            pos[1] = -pos[1]
            direction[1] = -direction[1]
        elif pos[1] > L:
            pos[1] = 2 * L - pos[1]
            direction[1] = -direction[1]

        # z 轴
        if pos[2] < 0:
            pos[2] = -pos[2]
            direction[2] = -direction[2]
        elif pos[2] > L:
            pos[2] = 2 * L - pos[2]
            direction[2] = -direction[2]

        # 重新归一化方向向量（避免数值误差累积）
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm

    def _generate_new_direction(self, old_dir):
        """
        根据翻转角度分布生成新的单位方向向量。

        方法：从正态分布 N(tumble_angle_mean, tumble_angle_std²) 采样偏转角 theta，
              从均匀分布 U(0, 2π) 采样方位角 phi，
              然后在垂直于旧方向的平面上旋转旧方向得到新方向。
        来源：Berg & Brown (1972); Saragosti et al. (2012) PLoS ONE 7:e35412.

        Parameters
        ----------
        old_dir : np.ndarray
            旧方向单位向量。

        Returns
        -------
        np.ndarray
            新的单位方向向量。
        """
        # 1. 采样偏转角 theta（度数 -> 弧度）
        theta_deg = np.random.normal(self.tumble_angle_mean, self.tumble_angle_std)
        theta = np.radians(theta_deg)

        # 2. 采样方位角 phi（弧度）
        phi = np.random.uniform(0, 2 * np.pi)

        # 3. 构造一组正交基，其中 e1 垂直于 old_dir，e2 = old_dir × e1
        #    (确保 e1, e2, old_dir 构成右手系)
        old_dir = old_dir / np.linalg.norm(old_dir)  # 确保单位向量

        # 选择一个不平行于 old_dir 的参考向量来构造垂直轴
        if abs(old_dir[0]) < 0.9:  # 避免与 [1,0,0] 平行
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        # 计算 e1 = old_dir × ref 归一化（垂直于 old_dir）
        e1 = np.cross(old_dir, ref)
        e1_norm = np.linalg.norm(e1)
        if e1_norm < 1e-12:
            # 如果叉积太小（几乎平行），换另一个参考向量
            ref = np.array([0.0, 0.0, 1.0])
            e1 = np.cross(old_dir, ref)
            e1_norm = np.linalg.norm(e1)
        e1 /= e1_norm

        # e2 = old_dir × e1 （自动归一化，因为 old_dir 和 e1 正交且为单位向量）
        e2 = np.cross(old_dir, e1)

        # 4. 计算新方向：在由 e1 和 e2 张成的平面内旋转 old_dir
        #    新方向 = cos(theta) * old_dir + sin(theta) * (cos(phi)*e1 + sin(phi)*e2)
        new_dir = (np.cos(theta) * old_dir +
                   np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2))

        # 归一化（理论上已经归一化，但数值误差可能累积）
        new_dir /= np.linalg.norm(new_dir)
        return new_dir