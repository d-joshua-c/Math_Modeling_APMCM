from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_forecast(csv_path: Path | str | None = None) -> pd.DataFrame:
    if csv_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        csv_path = repo_root / "data" / "processed" / "Q4_IMPTOTUS_forecast_without_tariff_48m.csv"
    df = pd.read_csv(csv_path)
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.set_index("Month")
    return df


def scale_to_minus_one_one(t: np.ndarray) -> np.ndarray:
    t_min, t_max = float(t.min()), float(t.max())
    if t_max == t_min:
        return np.zeros_like(t, dtype=float)
    return 2.0 * (t - t_min) / (t_max - t_min) - 1.0


def elasticity_series(n: int, coef: np.ndarray, const_after: int = 18) -> np.ndarray:
    t = np.arange(n, dtype=float)
    ts = scale_to_minus_one_one(t)
    e = np.polyval(coef, ts)
    k = min(const_after, n - 1)
    e_const = float(np.polyval(coef, ts[k]))
    if n > const_after:
        e[const_after:] = e_const
    return e


def apply_tariff_shock(df_fc: pd.DataFrame, tariff: float, e: np.ndarray) -> pd.DataFrame:
    out = df_fc.copy()
    factor = 1.0 + e * tariff
    for col in ["pred", "lower_50", "upper_50", "lower_80", "upper_80", "lower_95", "upper_95"]:
        if col in out.columns:
            out[f"{col}_shock"] = out[col].values * factor
    out["shock_factor"] = factor
    return out


def plot_result(df: pd.DataFrame):
    x = df.index
    plt.figure(figsize=(11, 5))
    if "pred" in df.columns:
        plt.plot(x, df["pred"].values, label="Baseline (no tariff)", color="#3366CC")
    if "pred_shock" in df.columns:
        plt.plot(x, df["pred_shock"].values, label="With tariff shock", color="#DC3912")
    plt.xlabel("Month")
    plt.ylabel("Imports (IMPTOTUS)")
    plt.title("Tariff Shock on U.S. Imports (poly-elasticity)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)


def main():
    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--tariff", type=float, default=0.18)
    parser.add_argument(
        "--coeffs",
        nargs="*",
        type=float,
        default=[
            -5.13270501e02,
            -3.08470765e02,
            1.47520760e03,
            7.75690156e02,
            -1.58688328e03,
            -6.89057429e02,
            7.91244308e02,
            2.54745797e02,
            -1.81655891e02,
            -3.32823612e01,
            1.53030554e01,
            1.98920378e-01,
        ],
    )
    parser.add_argument("--const_after", type=int, default=18)
    args = parser.parse_args()

    df_fc = load_forecast(args.csv)
    coef = np.asarray(args.coeffs, dtype=float)
    e = elasticity_series(len(df_fc), coef, const_after=args.const_after)
    df_out = apply_tariff_shock(df_fc, tariff=args.tariff, e=e)
    plot_result(df_out)


if __name__ == "__main__":
    main()