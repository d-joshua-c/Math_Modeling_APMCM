import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from statsmodels.tsa.stattools import adfuller
import sys

# 使同级模块可导入（将 code 目录加入路径）
sys.path.append(str(Path(__file__).resolve().parent))

# 引入模块化函数
from code.q4mod.io import load_data
from code.q4mod.modeling import choose_sarima, forecast
from code.q4mod.plotting import plot_forecast


# 加载函数已移至 q4mod.io


# 模型选择函数已移至 q4mod.modeling


# 预测函数已移至 q4mod.modeling


# 绘图函数已移至 q4mod.plotting


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

    fc_df = forecast(best_model, steps=48)
    print('预测前5条:')
    print(fc_df.head())

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / 'data' / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)
    fc_df.index.name = 'Month'
    out_path = out_dir / 'Q4_IMPTOTUS_forecast_without_tariff_48m.csv'
    fc_df.to_csv(out_path)

    # 绘图：英文，自动关闭5秒，文件名前缀使用程序名
    img_path = Path(__file__).with_suffix('.png')
    plot_forecast(y, fc_df, auto_close=True, save_path=img_path)


if __name__ == '__main__':
    # 依赖未安装时，请使用：
    # uv add pandas numpy matplotlib statsmodels scipy
    # 运行：
    # uv run python code/imports_forecast.py
    main()