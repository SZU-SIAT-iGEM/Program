# 微重力细菌趋化行为模拟 - 模块D：细菌代谢营养消耗模块
# 本模块负责计算细菌的营养消耗，并更新环境中的营养浓度场

# 导入配置文件中的参数
from config import MAX_CONSUMPTION_RATE, NUTRIENT_DIFFUSIVITY


def calculate_single_consumption(nutrient_concentration):
    """
    计算单个细菌的营养消耗量
    
    参数说明：
        nutrient_concentration (float): 细菌当前位置的营养浓度（0.0到1.0之间）
    
    返回值：
        float: 单个细菌本次消耗的营养量
    
    功能说明：
        根据当前位置的营养浓度，计算细菌消耗的营养量
        使用简单的饱和模型：营养浓度越高，消耗量越大，但有上限
    """
    
    # 确保营养浓度在合理范围内（0到1之间）
    if nutrient_concentration < 0:
        nutrient_concentration = 0
    elif nutrient_concentration > 1:
        nutrient_concentration = 1
    
    # 计算消耗量：使用简单的线性饱和模型
    # 消耗量 = 最大消耗速率 × 营养浓度
    # 这样当营养浓度为0时，消耗为0；当营养浓度为1时，消耗达到最大值
    consumption = MAX_CONSUMPTION_RATE * nutrient_concentration
    
    return consumption


def calculate_total_consumption(consumption_list):
    """
    计算所有细菌的总营养消耗量
    
    参数说明：
        consumption_list (list): 所有细菌的消耗量列表，例如 [0.3, 0.25, 0.4, ...]
    
    返回值：
        float: 所有细菌的总营养消耗量
    
    功能说明：
        将所有细菌的消耗量相加，得到总消耗量
    """
    
    # 初始化总消耗量为0
    total_consumption = 0.0
    
    # 遍历消耗列表，累加每个细菌的消耗量
    for consumption in consumption_list:
        total_consumption += consumption
    
    return total_consumption


def update_nutrient_field(nutrient_field, total_consumption, field_width, field_height):
    """
    更新环境营养浓度场（可选函数，供其他模块调用）
    
    参数说明：
        nutrient_field (list): 当前营养浓度场，二维列表
        total_consumption (float): 总营养消耗量
        field_width (int): 营养场的宽度（列数）
        field_height (int): 营养场的高度（行数）
    
    返回值：
        list: 更新后的营养浓度场
    
    功能说明：
        根据总消耗量，均匀地从营养场中扣除营养
        简化处理：将总消耗量平均分配到整个营养场
    """
    
    # 计算营养场中总的格子数
    total_cells = field_width * field_height
    
    # 计算每个格子需要扣除的营养量（平均分配）
    consumption_per_cell = total_consumption / total_cells
    
    # 遍历营养场，更新每个格子的营养浓度
    for i in range(field_height):
        for j in range(field_width):
            # 扣除营养，确保不小于0
            nutrient_field[i][j] -= consumption_per_cell
            if nutrient_field[i][j] < 0:
                nutrient_field[i][j] = 0
    
    return nutrient_field


# 示例使用（仅用于测试，实际使用时会被其他模块调用）
if __name__ == "__main__":
    print("=== 细菌代谢营养消耗模块测试 ===\n")
    
    # 测试1：计算单个细菌的消耗
    print("测试1：计算单个细菌的营养消耗")
    test_concentrations = [0.0, 0.3, 0.5, 0.8, 1.0]
    for conc in test_concentrations:
        consumption = calculate_single_consumption(conc)
        print(f"  营养浓度 {conc} -> 消耗量: {consumption:.4f}")
    
    # 测试2：计算总消耗
    print("\n测试2：计算所有细菌的总消耗")
    test_consumptions = [0.1, 0.2, 0.15, 0.25, 0.3]
    total = calculate_total_consumption(test_consumptions)
    print(f"  各细菌消耗量: {test_consumptions}")
    print(f"  总消耗量: {total:.4f}")
    
    # 测试3：更新营养场
    print("\n测试3：更新营养浓度场")
    field_width = 5
    field_height = 5
    nutrient_field = [[1.0 for _ in range(field_width)] for _ in range(field_height)]
    print(f"  更新前营养场左上角: {nutrient_field[0][:3]}")
    
    updated_field = update_nutrient_field(nutrient_field, total, field_width, field_height)
    print(f"  更新后营养场左上角: {updated_field[0][:3]}")
    
    print("\n=== 测试完成 ===")
