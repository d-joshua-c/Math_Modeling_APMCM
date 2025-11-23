# ==================================================
# 完整版：芯片数据随机森林+博弈论分析（适配你的数据）
# 运行顺序：直接执行，按提示输入列序号即可
# 保存路径：D:/__MXM/study/竞赛/数模/亚太杯/数据/问题3
# ==================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
warnings.filterwarnings('ignore')

# ========================
# 1. 全局配置（无需修改）
# ========================
# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 数据路径（你的实际路径，已固定）
DATA_PATH = r'D:/__MXM/study/竞赛/数模/亚太杯/数据/问题3/预处理后_三季度芯片数据.xlsx'
SAVE_DIR = os.path.dirname(DATA_PATH)  # 结果保存在数据同目录
print(f"📁 数据目录：{SAVE_DIR}")
print(f"📄 数据文件：{os.path.basename(DATA_PATH)}")


# ========================
# 2. 第一步：数据读取与彻底清理（解决字符串问题）
# ========================
print("\n" + "="*70)
print("【第一步：数据读取与清理（确保纯数值）】")
print("="*70)

# 2.1 读取原始数据
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ 文件不存在！请检查路径：{DATA_PATH}")
df_raw = pd.read_excel(DATA_PATH)
print(f"✅ 原始数据读取完成：{df_raw.shape}（行×列）")

# 2.2 清理注释行（删除含“注:”“为了防止倒卖”等文本的行）
comment_markers = ['注:', '说明:', '备注:', '为了防止倒卖', '数据来源:']
non_comment_mask = df_raw.apply(
    lambda row: not any(marker in str(cell) for marker in comment_markers for cell in row),
    axis=1
)
df_cleaned = df_raw[non_comment_mask].copy()
print(f"✅ 注释行清理：删除{len(df_raw)-len(df_cleaned)}行，剩余{len(df_cleaned)}行")

# 2.3 清理单元格文本（提取纯数值，解决“字符串转浮点数”错误）
def extract_numeric(cell):
    """从单元格提取纯数值，无法提取则返回NaN"""
    if pd.isna(cell):
        return np.nan
    if isinstance(cell, (int, float)):
        return cell
    cell_str = str(cell).strip()
    # 匹配纯数字（整数/小数/负数）
    num_match = re.search(r'^-?\d+(\.\d+)?$', cell_str)
    if num_match:
        return float(num_match.group())
    # 清理特殊符号（如“50+”→50，“≥14nm”→14）
    clean_str = re.sub(r'[^\d.-]', '', cell_str)
    if clean_str and re.match(r'^-?\d+(\.\d+)?$', clean_str):
        return float(clean_str)
    return np.nan

# 识别所有数值列（能提取出数值的列）
numeric_cols = []
for col in df_cleaned.columns:
    df_cleaned[col] = df_cleaned[col].apply(extract_numeric)
    # 超过50%的值为数值，则视为数值列
    if df_cleaned[col].notna().mean() > 0.5:
        numeric_cols.append(col)

# 保留数值列并填充NaN（用中位数，避免极端值影响）
df_final = df_cleaned[numeric_cols].copy()
for col in df_final.columns:
    df_final[col].fillna(df_final[col].median(), inplace=True)
df_final = df_final.astype(float)  # 强制转为float类型
print(f"✅ 数值列清理完成：共{len(df_final.columns)}个纯数值列")
print(f"   数值列列表：{df_final.columns.tolist()}")

# 保存清理后的数据（备用）
cleaned_save_path = os.path.join(SAVE_DIR, '清理后_纯数值数据.xlsx')
df_final.to_excel(cleaned_save_path, index=False)
print(f"✅ 清理后数据已保存：{os.path.basename(cleaned_save_path)}")


# ========================
# 3. 第二步：手动选择特征列与目标列（解决列名不匹配）
# ========================
print("\n" + "="*70)
print("【第二步：手动选择列（按提示输入序号）】")
print("="*70)

# 3.1 显示所有可用数值列，让用户选择“特征列”（影响因素）
print("\n📊 可用的数值列（请选择【特征列】——影响结果的因素，如关税率、补贴）：")
for i, col in enumerate(df_final.columns, 1):
    print(f"   {i:2d}. {col}")

# 交互选择特征列（输入序号，逗号分隔，如“1,3,5”）
while True:
    feat_input = input("\n请输入特征列序号（用英文逗号分隔，如1,2）：")
    try:
        # 解析输入的序号（转为0-based索引）
        feat_indices = [int(x.strip())-1 for x in feat_input.split(',')]
        feature_cols = [df_final.columns[i] for i in feat_indices]
        # 验证序号有效
        if len(feature_cols) == 0 or any(col not in df_final.columns for col in feature_cols):
            raise ValueError
        print(f"✅ 已选择特征列：{feature_cols}")
        break
    except:
        print("❌ 输入错误！请用英文逗号分隔序号（如1,2,3），且序号在上述列表中")

# 3.2 选择“目标列”（要分析的结果，如行业增长率、自给率）
print("\n🎯 可用的数值列（请选择【目标列】——要分析的结果，如增长率、自给率）：")
for i, col in enumerate(df_final.columns, 1):
    print(f"   {i:2d}. {col}")

while True:
    target_input = input("\n请输入目标列序号（仅选1个，如4）：")
    try:
        target_idx = int(target_input.strip())-1
        target_col = df_final.columns[target_idx]
        print(f"✅ 已选择目标列：{target_col}")
        break
    except:
        print("❌ 输入错误！请输入单个有效序号（如4）")


# ========================
# 4. 第三步：随机森林训练（提取特征权重）
# ========================
print("\n" + "="*70)
print("【第三步：随机森林模型训练】")
print("="*70)

# 4.1 准备训练数据
X = df_final[feature_cols]  # 特征矩阵
y = df_final[target_col]    # 目标变量
print(f"📥 训练数据：特征列{X.shape[1]}个，样本{X.shape[0]}条")

# 4.2 划分训练集/测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# 4.3 训练随机森林（参数优化，避免过拟合）
rf_model = RandomForestRegressor(
    n_estimators=100,    # 100棵决策树（稳定）
    max_depth=5,         # 限制树深（防止过拟合）
    min_samples_split=5, # 节点分裂最小样本数
    random_state=42      # 结果可复现
)
rf_model.fit(X_train, y_train)

# 4.4 模型评估（R²：越接近1，解释力越强）
y_pred = rf_model.predict(X_test)
r2_score_val = r2_score(y_test, y_pred)
print(f"📈 模型评估：测试集R² = {r2_score_val:.3f}（≥0.3即有效）")

# 4.5 提取特征重要性（核心输出，供博弈论使用）
feature_importance = pd.DataFrame({
    'Feature_Name': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n🔍 特征重要性排名（权重越高，影响越大）：")
for idx, row in feature_importance.iterrows():
    print(f"   {row.name+1:2d}. {row['Feature_Name']}：{row['Importance']:.3f}")

# 4.6 保存随机森林结果
rf_importance_path = os.path.join(SAVE_DIR, '随机森林_特征权重结果.xlsx')
feature_importance.to_excel(rf_importance_path, index=False)
print(f"\n✅ 特征权重已保存：{os.path.basename(rf_importance_path)}")

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature_Name'], feature_importance['Importance'], color='#4ECDC4')
plt.xlabel('特征重要性权重')
plt.title(f'随机森林特征重要性（目标列：{target_col}）')
plt.gca().invert_yaxis()  # 倒序：权重高的在上面
plt.tight_layout()
rf_plot_path = os.path.join(SAVE_DIR, '随机森林_特征重要性图.png')
plt.savefig(rf_plot_path)
print(f"✅ 特征重要性图已保存：{os.path.basename(rf_plot_path)}")


# ========================
# 5. 第四步：博弈论分析（基于随机森林权重）
# ========================
print("\n" + "="*70)
print("【第四步：博弈论纳什均衡求解】")
print("="*70)

# 5.1 读取随机森林权重结果
if not os.path.exists(rf_importance_path):
    raise FileNotFoundError(f"❌ 权重文件不存在：{rf_importance_path}")
importance_df = pd.read_excel(rf_importance_path)
print(f"📥 读取特征权重：共{len(importance_df)}个特征")

# 5.2 手动选择博弈相关特征（关税、补贴——博弈的核心策略）
print("\n⚖️  请选择博弈核心特征（从以下特征中选2个：关税相关、补贴相关）：")
for i, row in importance_df.iterrows():
    print(f"   {i+1:2d}. {row['Feature_Name']}（权重：{row['Importance']:.3f}）")

# 选择“关税相关特征”
while True:
    tariff_input = input("\n请输入【关税相关特征】序号（如1）：")
    try:
        tariff_idx = int(tariff_input.strip())-1
        tariff_feat = importance_df.iloc[tariff_idx]['Feature_Name']
        tariff_weight = importance_df.iloc[tariff_idx]['Importance']
        print(f"✅ 已选关税特征：{tariff_feat}（权重：{tariff_weight:.3f}）")
        break
    except:
        print("❌ 输入错误！请输入单个有效序号")

# 选择“补贴相关特征”
while True:
    subsidy_input = input("请输入【补贴相关特征】序号（如2）：")
    try:
        subsidy_idx = int(subsidy_input.strip())-1
        subsidy_feat = importance_df.iloc[subsidy_idx]['Feature_Name']
        subsidy_weight = importance_df.iloc[subsidy_idx]['Importance']
        print(f"✅ 已选补贴特征：{subsidy_feat}（权重：{subsidy_weight:.3f}）")
        break
    except:
        print("❌ 输入错误！请输入单个有效序号")

# 5.3 博弈模型设定（芯片行业场景）
# 博弈双方与策略
players = {
    '美国政府': ['关税策略', '补贴策略', '混合策略'],
    '全球市场/中国': ['买入美国芯片', '反制措施', '部分买入+部分反制']
}
us_strats = players['美国政府']
cn_strats = players['全球市场/中国']

# 芯片领域划分（按制程，贴合行业实际）
chip_domains = {
    '高端芯片': {'名称': '高端芯片（<10nm）', '经济权重占比': 0.3, '安全权重占比': 0.7},
    '中端芯片': {'名称': '中端芯片（10-28nm）', '经济权重占比': 0.5, '安全权重占比': 0.5},
    '低端芯片': {'名称': '低端芯片（>28nm）', '经济权重占比': 0.7, '安全权重占比': 0.3}
}

# 5.4 支付函数（计算美国得分，中国得分=10-美国得分）
def calc_payoff(domain, us_strat, cn_strat):
    """
    支付得分 = 经济贡献 + 安全贡献
    经济贡献 = 基础得分 × 策略调整 × 特征权重 × 领域经济占比
    安全贡献 = 基础得分 × 策略调整 × 特征权重 × 领域安全占比
    """
    # 1. 基础得分（按策略和领域设定）
    if us_strat == '关税策略':
        # 关税：低端经济收益高，高端安全收益低
        base_econ = 9 if '低端' in domain['名称'] else (6 if '中端' in domain['名称'] else 4)
        base_sec = 4 if '高端' in domain['名称'] else (6 if '中端' in domain['名称'] else 8)
        econ_w = tariff_weight  # 关税的经济权重
        sec_w = tariff_weight * 0.8  # 关税对安全的影响稍弱
    elif us_strat == '补贴策略':
        # 补贴：高端安全收益高，低端经济收益低
        base_econ = 4 if '低端' in domain['名称'] else (6 if '中端' in domain['名称'] else 9)
        base_sec = 9 if '高端' in domain['名称'] else (6 if '中端' in domain['名称'] else 4)
        econ_w = subsidy_weight * 0.8  # 补贴对经济的影响稍弱
        sec_w = subsidy_weight  # 补贴的安全权重
    else:  # 混合策略
        base_econ = 7
        base_sec = 7
        econ_w = (tariff_weight + subsidy_weight) / 2
        sec_w = (tariff_weight + subsidy_weight) / 2

    # 2. 中国策略调整系数（反制降分，买入升分）
    if cn_strat == '反制措施':
        adjust = 0.6  # 反制：美国得分降40%
    elif cn_strat == '买入美国芯片':
        adjust = 1.3  # 买入：美国得分升30%
    else:
        adjust = 0.9  # 部分买入：微降10%

    # 3. 计算最终得分
    econ_contrib = base_econ * adjust * econ_w * domain['经济权重占比']
    sec_contrib = base_sec * adjust * sec_w * domain['安全权重占比']
    return round(econ_contrib + sec_contrib, 2)

# 5.5 求解每个领域的纳什均衡
game_results = []
for domain_name, domain_info in chip_domains.items():
    # 构建支付矩阵（美国得分）
    payoff_mat = np.zeros((len(us_strats), len(cn_strats)))
    for i, us_strat in enumerate(us_strats):
        for j, cn_strat in enumerate(cn_strats):
            payoff_mat[i, j] = calc_payoff(domain_info, us_strat, cn_strat)

    # 标记最优策略：美国选最大得分，中国选最小得分（中国得分=10-美国得分）
    us_best = np.apply_along_axis(lambda x: x == x.max(), 0, payoff_mat)  # 美国最优（每列最大）
    cn_best = np.apply_along_axis(lambda x: x == x.min(), 1, payoff_mat)  # 中国最优（每行最小）
    eq_mask = us_best & cn_best  # 纳什均衡：双方均最优

    # 提取均衡策略
    equilibria = []
    for i in range(len(us_strats)):
        for j in range(len(cn_strats)):
            if eq_mask[i, j]:
                equilibria.append({
                    '美国策略': us_strats[i],
                    '中国策略': cn_strats[j],
                    '美国得分': payoff_mat[i, j],
                    '中国得分': round(10 - payoff_mat[i, j], 2)
                })

    game_results.append({
        '芯片领域': domain_info['名称'],
        '支付矩阵': payoff_mat,
        '纳什均衡': equilibria
    })
    print(f"\n📊 {domain_info['名称']}：")
    print(f"   支付矩阵（美国得分）：\n{payoff_mat.round(2)}")
    print(f"   纳什均衡策略（共{len(equilibria)}组）：")
    for eq in equilibria:
        print(f"     - 美国：{eq['美国策略']} | 中国：{eq['中国策略']} | 美得分：{eq['美国得分']}")

# 5.6 保存博弈结果
# 整理均衡结果
eq_final = []
for res in game_results:
    for eq in res['纳什均衡']:
        eq_final.append({
            '芯片领域': res['芯片领域'],
            '美国最优策略': eq['美国策略'],
            '中国最优策略': eq['中国策略'],
            '美国支付得分': eq['美国得分'],
            '中国支付得分': eq['中国得分']
        })
eq_df = pd.DataFrame(eq_final)

# 保存Excel
game_save_path = os.path.join(SAVE_DIR, '博弈论_纳什均衡结果.xlsx')
eq_df.to_excel(game_save_path, index=False)
print(f"\n✅ 博弈结果已保存：{os.path.basename(game_save_path)}")

# 绘制支付矩阵热力图
plt.figure(figsize=(18, 5))
for i, res in enumerate(game_results):
    plt.subplot(1, 3, i+1)
    sns.heatmap(
        res['支付矩阵'],
        annot=True, fmt='.2f',
        xticklabels=cn_strats,
        yticklabels=us_strats,
        cmap='YlOrRd',
        cbar=False,
        annot_kws={'fontsize': 9}
    )
    # 标注纳什均衡位置（红色星号）
    for j in range(len(us_strats)):
        for k in range(len(cn_strats)):
            if eq_mask[j, k]:
                plt.text(k+0.5, j+0.5, '*', ha='center', va='center', color='red', fontsize=16)
    plt.title(res['芯片领域'], fontsize=10)
    plt.xlabel('中国策略', fontsize=9)
    plt.ylabel('美国策略', fontsize=9)

game_plot_path = os.path.join(SAVE_DIR, '博弈论_支付矩阵热力图.png')
plt.tight_layout()
plt.savefig(game_plot_path)
print(f"✅ 博弈热力图已保存：{os.path.basename(game_plot_path)}")


# ========================
# 6. 最终总结
# ========================
print("\n" + "="*70)
print("【最终结果总结】")
print("="*70)
print(f"📁 所有结果保存在：{SAVE_DIR}")
print(f"1. 清理后数据：清理后_纯数值数据.xlsx")
print(f"2. 随机森林结果：随机森林_特征权重结果.xlsx、随机森林_特征重要性图.png")
print(f"3. 博弈论结果：博弈论_纳什均衡结果.xlsx、博弈论_支付矩阵热力图.png")
print(f"\n✅ 完整分析完成！可直接使用上述文件进行报告撰写。")