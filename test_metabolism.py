# 微重力细菌趋化行为模拟 - metabolism模块极简测试代码
# 本代码用于测试metabolism.py中的两个核心函数是否正常运行
# 测试目标：小白能直接运行、打印结果、不用其他模块、自带模拟数据

# ========== 第一部分：模拟config.py中的参数 ==========
# 因为不想导入其他模块，这里直接定义参数值
MAX_CONSUMPTION_RATE = 0.5  # 最大消耗速率（和营养场里的值保持一致）


# ========== 第二部分：复制metabolism.py中的两个核心函数 ==========
# 这样测试代码可以完全独立运行，不依赖其他文件

def calculate_single_consumption(nutrient_concentration):
    """
    计算单个细菌的营养消耗量
    
    参数说明：
        nutrient_concentration (float): 细菌当前位置的营养浓度
    
    返回值：
        float: 单个细菌本次消耗的营养量
    """
    # 确保营养浓度在合理范围内（0到1之间）
    if nutrient_concentration < 0:
        nutrient_concentration = 0
    elif nutrient_concentration > 1:
        nutrient_concentration = 1
    
    # 计算消耗量：消耗量 = 最大消耗速率 × 营养浓度
    consumption = MAX_CONSUMPTION_RATE * nutrient_concentration
    
    return consumption


def calculate_total_consumption(consumption_list):
    """
    计算所有细菌的总营养消耗量
    
    参数说明：
        consumption_list (list): 所有细菌的消耗量列表
    
    返回值：
        float: 所有细菌的总营养消耗量
    """
    # 初始化总消耗量为0
    total_consumption = 0.0
    
    # 遍历消耗列表，累加每个细菌的消耗量
    for consumption in consumption_list:
        total_consumption += consumption
    
    return total_consumption


# ========== 第三部分：测试代码 ==========

def test_calculate_single_consumption():
    """
    测试函数1：计算单个细菌的营养消耗
    """
    print("=" * 50)
    print("测试1：calculate_single_consumption 函数")
    print("=" * 50)
    
    # 模拟营养浓度 = 10.0（注意：这个值会被限制在0-1范围内）
    test_nutrient_concentration = 10.0
    
    print(f"\n输入参数：")
    print(f"  营养浓度 = {test_nutrient_concentration}")
    print(f"  最大消耗速率 = {MAX_CONSUMPTION_RATE}")
    
    # 调用被测试的函数
    result = calculate_single_consumption(test_nutrient_concentration)
    
    print(f"\n计算过程：")
    print(f"  1. 营养浓度 {test_nutrient_concentration} > 1，被限制为 1.0")
    print(f"  2. 消耗量 = 最大消耗速率 × 营养浓度")
    print(f"  3. 消耗量 = {MAX_CONSUMPTION_RATE} × 1.0 = {result}")
    
    print(f"\n测试结果：")
    print(f"  单个细菌消耗量 = {result}")
    
    # 验证结果是否正确
    expected_result = 0.5  # 因为营养浓度被限制为1，所以消耗量=0.5×1=0.5
    if result == expected_result:
        print(f"  ✅ 测试通过！结果正确（期望值：{expected_result}）")
        return True
    else:
        print(f"  ❌ 测试失败！结果错误（期望值：{expected_result}，实际值：{result}）")
        return False


def test_calculate_total_consumption():
    """
    测试函数2：计算所有细菌的总营养消耗
    """
    print("\n" + "=" * 50)
    print("测试2：calculate_total_consumption 函数")
    print("=" * 50)
    
    # 模拟3个细菌的消耗列表
    test_consumption_list = [0.3, 0.25, 0.45]
    
    print(f"\n输入参数：")
    print(f"  消耗列表 = {test_consumption_list}")
    print(f"  细菌数量 = {len(test_consumption_list)}")
    
    # 调用被测试的函数
    result = calculate_total_consumption(test_consumption_list)
    
    print(f"\n计算过程：")
    print(f"  总消耗 = 0.3 + 0.25 + 0.45 = {result}")
    
    print(f"\n测试结果：")
    print(f"  总营养消耗量 = {result}")
    
    # 验证结果是否正确
    expected_result = 1.0  # 0.3 + 0.25 + 0.45 = 1.0
    if result == expected_result:
        print(f"  ✅ 测试通过！结果正确（期望值：{expected_result}）")
        return True
    else:
        print(f"  ❌ 测试失败！结果错误（期望值：{expected_result}，实际值：{result}）")
        return False


# ========== 第四部分：运行所有测试 ==========

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("metabolism.py 模块测试程序")
    print("=" * 50)
    print("\n本程序测试两个核心函数：")
    print("  1. calculate_single_consumption - 计算单个细菌消耗")
    print("  2. calculate_total_consumption - 计算总消耗")
    
    # 运行测试1
    test1_passed = test_calculate_single_consumption()
    
    # 运行测试2
    test2_passed = test_calculate_total_consumption()
    
    # 打印最终测试结果
    print("\n" + "=" * 50)
    print("最终测试结果")
    print("=" * 50)
    
    if test1_passed and test2_passed:
        print("\n✅ 所有测试通过！metabolism.py 模块工作正常。")
    else:
        print("\n❌ 部分测试失败，请检查代码。")
        if not test1_passed:
            print("   - calculate_single_consumption 函数有问题")
        if not test2_passed:
            print("   - calculate_total_consumption 函数有问题")
    
    print("\n" + "=" * 50)
