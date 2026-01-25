# Program
作为SZU-SIAT-iGEM2026的程序开发仓库，存放所有干实验建模、编程相关的内容。由建模和程序的负责人自行管理。
## 📁 07_程序开发

### 用途说明

存放 **所有干实验、建模、编程** 相关的内容。本文件夹结构由程序组自行管理，以下为建议结构。

### 建议目录结构（例，自行编辑）

```
07_程序开发/
├── 📁 建模/
│   ├── 📁 生长预测模型A/
│   │   ├── 📁 Note
│   │   ├── 📄 README.md
│   │   ├── 📄 model.py
│   │   ├── 📁 data/
│   │   ├── 📁 results/
│   │   └── ...
│   ├── 📁 代谢通量模型B/
│   └── 📁 启动子预测模型C/
├── 📁 数据分析/
│   ├── 📁 实验数据处理/
│   └── 📁 统计分析/
├── 📁 软件工具/
│   ├── 📁 Tool_序列分析工具/
│   └── ...
├── 📁 Wiki开发/
│   ├── 📄 README.md
│   ├── 📁 src/
│   ├── 📁 assets/
│   └── ...
├── 📁 文档/
│   ├── 📄 环境配置指南.md
│   ├── 📄 代码规范.md
│   └── ...
├── 📄 README.md
└── 📄 requirements.txt
```

### 操作指南

#### 项目结构规范

- 每个模型按功能分类

- 每个独立的代码项目应包含：

```
Project_项目名称/
├── 📄 README.md        # 项目说明、使用方法
├── 📄 requirements.txt # Python 依赖（如适用）
├── 📁 src/             # 源代码
├── 📁 data/            # 数据文件
├── 📁 docs/            # 文档
├── 📁 tests/           # 测试代码
└── 📁 results/         # 输出结果
```

#### README.md 最低要求

每个项目的 README 至少包含：

```markdown
# 项目名称

## 简介
一句话描述这个项目做什么

## 依赖环境
- Python 3.9+
- 其他依赖...

## 安装方法
```bash
pip install -r requirements.txt
```

## 使用方法
```bash
python main.py --input data.csv
```

## 输入输出
- 输入：xxx
- 输出：xxx

## 负责人
@username
```

#### 代码提交规范

| Commit 类型 | 格式 | 示例 |
|-------------|------|------|
| 新功能 | `[feat] 描述` | `[feat] 添加序列比对功能` |
| 修复 | `[fix] 描述` | `[fix] 修复内存溢出问题` |
| 文档 | `[docs] 描述` | `[docs] 更新使用说明` |
| 重构 | `[refactor] 描述` | `[refactor] 优化算法性能` |
| 测试 | `[test] 描述` | `[test] 添加单元测试` |

### .gitignore 建议

在文件夹中创建 `.gitignore` 文件：

```gitignore
# Python
__pycache__/
*.py[cod]
.env
venv/

# 数据文件（大文件）
*.csv
*.xlsx
!example_*.csv

# 系统文件
.DS_Store
Thumbs.db

# IDE
.idea/
.vscode/
*.swp
```

### 自由扩展

程序组可完全自由组织内部结构，但建议：
- 保持清晰的项目划分
- 每个项目都有 README
- 重要代码有注释
- 大数据文件使用 `.gitignore` 排除或使用 Git LFS
