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


def elasticity_poly_with_decay(n: int, coef: np.ndarray, const_after: int, decay_rate: float) -> np.ndarray:
    t = np.arange(n, dtype=float)
    ts = scale_to_minus_one_one(t)
    e = np.polyval(coef, ts)
    if n > const_after:
        k = max(const_after - 1, 0)
        e0 = float(e[k])
        idx = np.arange(const_after, n, dtype=float)
        e[const_after:] = e0 * np.exp(-decay_rate * (idx - k))
    return e


def add_gaussian_noise(n: int, std: float, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, std, n)


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
    plt.title("Tariff Shock on U.S. Imports (poly + noise + decay)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)


def run(
    csv_path: Path | str | None = None,
    tariff: float = 0.18,
    coeffs: list[float] | np.ndarray | None = None,
    const_after: int = 18,
    decay_rate: float = 0.1,
    noise_std: float = 0.05,
    seed: int | None = None,
) -> pd.DataFrame:
    df_fc = load_forecast(csv_path)
    if coeffs is None:
        coeffs = [
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
        ]
    coef = np.asarray(coeffs, dtype=float)
    n = len(df_fc)
    e_base = elasticity_poly_with_decay(n, coef, const_after=const_after, decay_rate=decay_rate)
    noise = add_gaussian_noise(n, std=noise_std, seed=seed)
    e = e_base + noise
    df_out = apply_tariff_shock(df_fc, tariff=tariff, e=e)
    plot_result(df_out)
    return df_out


def main():
    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--tariff", type=float, default=0.18)
    parser.add_argument("--const_after", type=int, default=18)
    parser.add_argument("--decay_rate", type=float, default=0.1)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
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
    args = parser.parse_args()
    run(
        csv_path=args.csv,
        tariff=args.tariff,
        coeffs=args.coeffs,
        const_after=args.const_after,
        decay_rate=args.decay_rate,
        noise_std=args.noise_std,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()