# Math_Modeling_APMCM

###### v-1.0.0: 搞定了整体文件夹的结构

```text fold:项目文件结构
APMCM_2025_ProblemC/
├── .venv/                  # uv 自动生成
├── .git/                   # git init 生成
├── pyproject.toml          # uv 生成
├── .gitignore              # 见下文配置
├── README.md               # 记录思路
├── data/                   # 存放数据
│   ├── raw/                # 原始 Excel/CSV (只读，不改)
│   └── processed/          # Python 清洗后导出的 JSON/CSV
├── code/                   # Python 脚本
│   ├── _notebooks/         # 你的草稿本 (Jupyter)
│   ├── cleaning.py         # 数据清洗脚本
│   ├── q1_model.py         # Q1 模型
│   └── utils.py            # 通用函数
└── viz/                    # 前端可视化
    ├── lib/
    │   └── echarts.min.js  # 下载好放在本地，防止断网
    ├── css/
    ├── q1_sankey.html      # 桑基图模板
    ├── q2_line.html        # 折线图模板
    └── q5_heatmap.html     # 热力图模板
```

###### v-1.0.1: 构建开发工具流
1. 更新之前的`.trae`文件: 删掉了非常细节的内容, 只保留了uv和powershell的使用须知
1. 采用`uv`初始化了项目, 下载了相关的包, 添加了`.gitignore`

