import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_elasticities(csv_path: Path | str | None = None) -> pd.Series:
    df_path: Path
    if csv_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        df_path = repo_root / "data" / "processed" / "Q4.trade_war_2018_2019.csv"
    else:
        df_path = Path(csv_path)
    df = pd.read_csv(df_path)
    y = df["real_elasticities"].astype(float).dropna().reset_index(drop=True)
    return y


def scale_to_minus_one_one(t: np.ndarray) -> np.ndarray:
    t_min, t_max = float(t.min()), float(t.max())
    if t_max == t_min:
        return np.zeros_like(t, dtype=float)
    return 2.0 * (t - t_min) / (t_max - t_min) - 1.0


def fit_and_show(y: pd.Series, degree: int = 12) -> np.ndarray:
    n = len(y)
    if n < degree + 2:
        raise ValueError(f"数据点过少: {n} < {degree + 2}")
    t = np.arange(n, dtype=float)
    t_scaled = scale_to_minus_one_one(t)
    coef = np.polyfit(t_scaled, y.values, deg=degree)

    t_dense = np.linspace(t_scaled.min(), t_scaled.max(), 400)
    y_dense = np.polyval(coef, t_dense)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(n), y.values, "o", label="Observed elasticity", color="#3366CC")
    plt.plot(
        np.linspace(0, n - 1, len(t_dense)),
        y_dense,
        "-",
        label=f"{degree}th-degree polynomial fit",
        color="#DC3912",
    )
    plt.xlabel("Month index (0 = 2018-07)")
    plt.ylabel("Trade elasticity")
    plt.title("U.S. Import Trade Elasticity (2018-2019) — Polynomial Fit")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    return coef


def run(csv_path: Path | str | None = None, degree: int = 12) -> np.ndarray:
    y = load_elasticities(csv_path)
    return fit_and_show(y, degree=degree)