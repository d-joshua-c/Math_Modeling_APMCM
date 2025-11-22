import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


def load_data(csv_path: Path) -> pd.Series:
    # 读取IMPTOTUS数据，筛选到2025-04（关税前）
    df = pd.read_csv(csv_path)
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.sort_values('observation_date')
    df = df[df['observation_date'] <= pd.Timestamp('2025-04-01')]
    df = df.set_index('observation_date').asfreq('MS')
    y = df['IMPTOTUS'].astype(float)
    return y


def choose_sarima(y_log: pd.Series):
    # 简单网格搜索选择SARIMA(p,1,q)(P,1,Q,12)（月度季节性）
    p_vals = [0, 1, 2]
    q_vals = [0, 1, 2]
    P_vals = [0, 1, 2]
    Q_vals = [0, 1]
    best_aic = np.inf
    best_cfg = None
    best_model = None
    for p in p_vals:
        for q in q_vals:
            for P in P_vals:
                for Q in Q_vals:
                    try:
                        model = SARIMAX(
                            y_log,
                            order=(p, 1, q),
                            seasonal_order=(P, 1, Q, 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        res = model.fit(disp=False)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_cfg = (p, 1, q, P, 1, Q, 12)
                            best_model = res
                    except Exception:
                        pass
    return best_cfg, best_model, best_aic


def forecast_48(y: pd.Series, best_model) -> pd.DataFrame:
    # 未来48个月预测（log尺度→反对数回到原尺度），包含多层次置信区间（50/80/95）
    h = 48
    fc_res = best_model.get_forecast(steps=h)
    pred_log = fc_res.predicted_mean
    pred = np.exp(pred_log)

    # 生成多置信水平的区间（alpha越小，区间越宽）
    ci50_log = fc_res.conf_int(alpha=0.5)
    ci80_log = fc_res.conf_int(alpha=0.2)
    ci95_log = fc_res.conf_int(alpha=0.05)

    out = pd.DataFrame({'pred': np.exp(pred_log)})
    out['lower_50'] = np.exp(ci50_log.iloc[:, 0])
    out['upper_50'] = np.exp(ci50_log.iloc[:, 1])
    out['lower_80'] = np.exp(ci80_log.iloc[:, 0])
    out['upper_80'] = np.exp(ci80_log.iloc[:, 1])
    out['lower_95'] = np.exp(ci95_log.iloc[:, 0])
    out['upper_95'] = np.exp(ci95_log.iloc[:, 1])

    return out


def plot_result(y: pd.Series, fc_df: pd.DataFrame):
    # 可视化历史与预测（多层次置信区间渐变显示）
    plt.figure(figsize=(10, 5))
    plt.plot(y.index, y, label='Historical Imports (million USD)')
    plt.plot(fc_df.index, fc_df['pred'], label='Forecast', color='red')

    # 渐变：内层更深、外层更浅（红系）
    plt.fill_between(fc_df.index, fc_df['lower_50'], fc_df['upper_50'], color='#d62728', alpha=0.35, label='Confidence 50%')
    plt.fill_between(fc_df.index, fc_df['lower_80'], fc_df['upper_80'], color='#ff7f0e', alpha=0.25, label='Confidence 80%')
    plt.fill_between(fc_df.index, fc_df['lower_95'], fc_df['upper_95'], color='#ff9896', alpha=0.18, label='Confidence 95%')

    plt.title('US Goods Imports: Time Series Forecast (using data up to 2025-04)')
    plt.xlabel('Month')
    plt.ylabel('Million USD')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # 命令行参数：允许自定义CSV路径；默认使用仓库内路径
    parser = ArgumentParser()
    parser.add_argument('--csv', type=str, default=None)
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        csv_path = repo_root / 'data' / 'raw' / 'Q4' / 'IMPTOTUS.csv'

    print(f'使用数据文件: {csv_path}')

    # 加载数据与基本检验
    y = load_data(csv_path)
    y_log = np.log(y)
    adf_stat, adf_p, *_ = adfuller(y_log.dropna())
    print(f'ADF统计量: {adf_stat:.4f}  p值: {adf_p:.6f}')

    # 选择并拟合最优SARIMA
    best_cfg, best_model, best_aic = choose_sarima(y_log)
    print(f'最佳模型: {best_cfg}  AIC: {best_aic:.2f}')

    # 预测未来48个月
    fc_df = forecast_48(y, best_model)
    print('预测前5条:')
    print(fc_df.head())

    # 绘图展示
    plot_result(y, fc_df)


if __name__ == '__main__':
    # 依赖未安装时，请使用：
    # uv add pandas numpy matplotlib statsmodels scipy
    # 运行：
    # uv run python code/imports_forecast.py
    main()