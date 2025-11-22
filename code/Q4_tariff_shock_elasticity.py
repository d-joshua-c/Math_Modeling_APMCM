import time
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 读取预测数据（无关税情景），并设置时间索引
def load_forecast(csv_path: Path) -> pd.DataFrame:
    # 读取CSV并解析日期
    df = pd.read_csv(csv_path, parse_dates=["Month"])
    df = df.set_index("Month").sort_index()
    # 基准预测值列为 pred（单位：百万美元）
    if "pred" not in df.columns:
        raise ValueError("CSV缺少列 'pred'，请确认文件格式")
    return df


# 动态弹性函数：e(t) = final - (final - initial) * exp(-speed * t)
def get_elasticity(t: np.ndarray, initial: float, final: float, speed: float) -> np.ndarray:
    # t 从0开始的月份序号（np.arange）
    return final - (final - initial) * np.exp(-speed * t)


# 根据弹性与关税变化，计算冲击后的进口量与关税收入
def compute_shock(
    df: pd.DataFrame,
    old_rate: float,
    new_rate: float,
    initial_elasticity: float,
    final_elasticity: float,
    speed: float,
) -> pd.DataFrame:
    # 价格变化比例（假设关税完全传导）
    delta_rate = new_rate - old_rate
    price_change_pct = delta_rate / (1.0 + old_rate)

    # 月份序列（0..N-1）
    t = np.arange(len(df))
    e_t = get_elasticity(t, initial_elasticity, final_elasticity, speed)

    baseline = df["pred"].to_numpy(dtype=float)
    shocked_volume = baseline * (1.0 + e_t * price_change_pct)
    revenue = shocked_volume * new_rate

    out = df.copy()
    out["baseline"] = baseline
    out["elasticity"] = e_t
    out["shocked_volume"] = shocked_volume
    out["revenue"] = revenue
    return out


# 绘图（英文），左轴显示进口量（基准/冲击），右轴显示关税收入；图像5秒后自动关闭
def plot_result(df: pd.DataFrame, save_dir: Path, program_name: str) -> Path:
    plt.figure(figsize=(10, 5))

    # 左轴：进口量（百万美元）
    ax_left = plt.gca()
    ax_left.plot(df.index, df["baseline"], label="Baseline Imports", color="#1f77b4")
    ax_left.plot(df.index, df["shocked_volume"], label="Shocked Imports", color="#d62728")
    ax_left.set_xlabel("Month")
    ax_left.set_ylabel("Million USD")

    # 右轴：关税收入（百万美元）
    ax_right = ax_left.twinx()
    ax_right.plot(df.index, df["revenue"], label="Tariff Revenue", color="#2ca02c")
    ax_right.set_ylabel("Tariff Revenue (Million USD)")

    # 合并图例（两轴）
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    plt.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")

    plt.title("US Goods Imports Tariff Shock (Elasticity)")
    plt.grid(True, axis="both", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = save_dir / f"{program_name}.png"
    plt.savefig(out_path, dpi=160)
    plt.show(block=False)
    time.sleep(5)
    plt.close()
    return out_path


def main():
    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="预测CSV路径（默认读取仓库processed目录）")
    parser.add_argument("--old_rate", type=float, default=0.03, help="旧税率")
    parser.add_argument("--new_rate", type=float, default=0.13, help="新税率")
    parser.add_argument("--initial_elasticity", type=float, default=0.91, help="短期弹性（t=0）")
    parser.add_argument("--final_elasticity", type=float, default=-0.49, help="长期弹性（t->∞）")
    parser.add_argument("--speed", type=float, default=0.03, help="调整速度")
    args = parser.parse_args()

    # 默认CSV路径：data/processed/Q4_IMPTOTUS_forecast_without_tariff_48m.csv
    if args.csv:
        csv_path = Path(args.csv)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        csv_path = repo_root / "data" / "processed" / "Q4_IMPTOTUS_forecast_without_tariff_48m.csv"

    df_fc = load_forecast(csv_path)

    df_out = compute_shock(
        df_fc,
        old_rate=args.old_rate,
        new_rate=args.new_rate,
        initial_elasticity=args.initial_elasticity,
        final_elasticity=args.final_elasticity,
        speed=args.speed,
    )

    # 保存图像到同目录（程序名作为前缀）
    program_name = Path(__file__).stem
    save_dir = Path(__file__).resolve().parent
    out_img = plot_result(df_out, save_dir, program_name)

    print(f"Image saved: {out_img}")


if __name__ == "__main__":
    # 依赖：pandas numpy matplotlib
    # 运行：uv run python code/Q4_tariff_shock_elasticity.py
    main()