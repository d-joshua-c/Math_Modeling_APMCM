import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------
# 1. 数据加载与预处理（简化，移除标准化）
# ------------------------------------------------------
def load_and_preprocess_data(file_path="trade_effect_data.csv"):
    """加载数据并进行基本预处理（树模型无需标准化）"""
    try:
        df = pd.read_csv(file_path)
        print("数据加载成功！")
    except FileNotFoundError:
        print("未找到数据文件，使用示例数据演示...")
        np.random.seed(42)
        df = pd.DataFrame({
            'GDP_Reporter_BnUSD': np.random.rand(60) * 1000 + 500,
            'GDP_Partner_BnUSD': np.random.rand(60) * 1000 + 500,
            'Distance_km': np.random.rand(60) * 10000 + 1000,
            'Tariff_Rate': np.random.rand(60) * 20,
            'Soybean_Price_USD_Ton': np.random.rand(60) * 500 + 3000,
            'wtqty': 5000 + 0.8*np.random.rand(60)*10000 + 0.3*np.random.randn(60)*1000 + 0.5*np.random.rand(60)*5000,
            'value': 10000 + 1.2*np.random.rand(60)*15000 + 0.5*np.random.randn(60)*2000 + 0.4*np.random.rand(60)*8000
        })

    # 特征列（无需加权，直接使用原始特征）
    feature_cols = ['GDP_Reporter_BnUSD', 'GDP_Partner_BnUSD', 'Distance_km',
                    'Tariff_Rate', 'Soybean_Price_USD_Ton']
    target_cols = ['wtqty', 'value']

    # 检查特征列是否存在
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        print(f"错误：数据中缺少特征列 {missing_features}，请检查数据文件！")
        return None

    # 提取特征和目标变量
    X = df[feature_cols].copy()
    y = df[target_cols].copy()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"特征数量：{X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, feature_cols, target_cols

# ------------------------------------------------------
# 2. 随机森林回归
# ------------------------------------------------------
def random_forest_predict(X_train, y_train_target, X_test):
    """随机森林回归预测"""
    rf = RandomForestRegressor(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train_target)
    return rf.predict(X_test)

# ------------------------------------------------------
# 3. 梯度提升回归
# ------------------------------------------------------
def gradient_boosting_predict(X_train, y_train_target, X_test):
    """梯度提升回归预测"""
    gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train_target)
    return gb.predict(X_test)

# ------------------------------------------------------
# 4. 优化组合权重（最小化MSE）
# ------------------------------------------------------
def find_optimal_weights(models_predictions, y_test_target):
    """使用数值优化找到最小化MSE的组合权重"""
    n_models = len(models_predictions)
    predictions_matrix = np.column_stack(models_predictions)

    # 目标函数：计算MSE
    def objective(weights):
        combined_pred = np.dot(predictions_matrix, weights)
        return mean_squared_error(y_test_target, combined_pred)

    # 初始权重（等权重）
    initial_weights = np.ones(n_models) / n_models

    # 约束条件：权重之和为1，且非负
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, None) for _ in range(n_models)]

    # 执行优化
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        print("优化失败，使用等权重代替！")
        return initial_weights

    return result.x

# ------------------------------------------------------
# 5. 组合预测（使用优化权重）
# ------------------------------------------------------
def combined_prediction_optimized(models_predictions, y_test_target):
    """使用优化权重进行组合预测"""
    optimal_weights = find_optimal_weights(models_predictions, y_test_target)
    predictions_matrix = np.column_stack(models_predictions)
    y_pred_combined = np.dot(predictions_matrix, optimal_weights)
    return y_pred_combined, optimal_weights

# ------------------------------------------------------
# 6. 主函数：完整流程
# ------------------------------------------------------
def main():
    data = load_and_preprocess_data()
    if data is None:
        return
    X_train, X_test, y_train, y_test, feature_cols, target_cols = data
    n_test = len(y_test)

    all_y_pred_combined = {}
    all_opt_weights = {}

    for target in target_cols:
        print(f"\n{'='*25} 处理目标变量: {target} {'='*25}")
        y_train_target = y_train[target]
        y_test_target = y_test[target]

        # 执行各模型预测（仅随机森林和梯度提升）
        print("正在运行模型预测...")
        y_pred_rf = random_forest_predict(X_train, y_train_target, X_test)
        y_pred_gb = gradient_boosting_predict(X_train, y_train_target, X_test)

        # 模型列表
        models_predictions = [y_pred_rf, y_pred_gb]
        model_names = ['随机森林', '梯度提升']

        # 优化组合权重并预测
        print("正在优化组合权重...")
        y_pred_combined, opt_weights = combined_prediction_optimized(models_predictions, y_test_target)

        all_y_pred_combined[target] = y_pred_combined
        all_opt_weights[target] = opt_weights

        # 输出结果
        print("\n--- 各模型评估结果 ---")
        for name, y_pred, w in zip(model_names, models_predictions, opt_weights):
            mse = mean_squared_error(y_test_target, y_pred)
            r2 = r2_score(y_test_target, y_pred)
            print(f"{name:<10} - MSE: {mse:>8.2f}, R²: {r2:>6.4f}, 权重: {w:.4f}")

        # 组合模型结果
        mse_combined = mean_squared_error(y_test_target, y_pred_combined)
        r2_combined = r2_score(y_test_target, y_pred_combined)
        print(f"\n{'组合模型':<10} - MSE: {mse_combined:>8.2f}, R²: {r2_combined:>6.4f}")

    # 可视化
    print("\n=== 组合预测结果可视化 ===")
    n_targets = len(target_cols)
    fig, axes = plt.subplots(n_targets, 1, figsize=(14, 6 * n_targets), squeeze=False)

    for i, target in enumerate(target_cols):
        ax = axes[i, 0]
        ax.plot(range(n_test), y_test[target].values, 'b-', label='真实值', linewidth=3, alpha=0.7)
        ax.plot(range(n_test), all_y_pred_combined[target], 'r-', label='优化组合预测值', linewidth=2)
        ax.set_xlabel('测试集样本索引')
        ax.set_ylabel(target)
        ax.set_title(f'优化组合模型预测结果: {target}')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()