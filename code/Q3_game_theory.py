from __future__ import annotations

from pathlib import Path
from argparse import ArgumentParser

import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================
# 数据加载与清理
# =============================
def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """
    加载问题三原始数据
    - 默认读取 data/raw/Q3/Q3_Data_English_Headers.csv
    - 若传入为 .xlsx 则自动用 Excel 方式读取
    """
    if path is None:
        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "data" / "raw" / "Q3" / "Q3_Data_English_Headers.csv"
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"未找到数据文件: {p}")
    if p.suffix.lower() == ".xlsx":
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    return df


def extract_numeric(cell) -> float | np.nan:
    """
    提取单元格中的纯数值, 失败返回 NaN
    """
    if pd.isna(cell):
        return np.nan
    if isinstance(cell, (int, float)):
        return float(cell)
    s = str(cell).strip()
    m = re.search(r"^-?\d+(\.\d+)?$", s)
    if m:
        return float(m.group())
    s2 = re.sub(r"[^\d.-]", "", s)
    if s2 and re.match(r"^-?\d+(\.\d+)?$", s2):
        return float(s2)
    return np.nan


def clean_numeric(df: pd.DataFrame, min_numeric_ratio: float = 0.5) -> pd.DataFrame:
    """
    将可解析为数值的列转为 float, 保留数值占比>阈值的列, 用中位数填充缺失
    """
    out = df.copy()
    numeric_cols: list[str] = []
    for col in out.columns:
        out[col] = out[col].apply(extract_numeric)
        if out[col].notna().mean() > min_numeric_ratio:
            numeric_cols.append(col)
    out = out[numeric_cols].copy()
    for col in out.columns:
        med = out[col].median()
        out[col] = out[col].fillna(med).astype(float)
    return out


# =============================
# 特征重要性计算
# =============================
def pick_target_column(df_num: pd.DataFrame, target_hint: str | None = None) -> str:
    """
    自动选择目标列
    - 若提供 target_hint, 优先匹配包含该关键词的列
    - 否则根据关键词集合与方差选择最可能的结果列
    """
    cols = list(df_num.columns)
    if target_hint:
        mask = [c for c in cols if target_hint.lower() in c.lower()]
        if mask:
            return mask[0]
    keys = ["growth", "rate", "index", "self", "supply", "value", "产值", "增速", "自给"]
    candidates = [c for c in cols if any(k in c.lower() for k in keys)]
    if candidates:
        # 选方差最大的候选
        var_s = {c: float(np.var(df_num[c].values)) for c in candidates}
        return max(var_s, key=var_s.get)
    # 兜底: 选方差最大的一列
    var_all = {c: float(np.var(df_num[c].values)) for c in cols}
    return max(var_all, key=var_all.get)


def compute_feature_importance(
    df_num: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """
    计算特征重要性
    - 优先使用 sklearn RandomForestRegressor
    - 无 sklearn 时回退为与目标的绝对相关系数
    返回 DataFrame[Feature_Name, Importance]
    """
    Xcols = [c for c in df_num.columns if c != target_col]
    X = df_num[Xcols]
    y = df_num[target_col]
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_split=5, random_state=42)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        rf.fit(Xtr, ytr)
        imp = rf.feature_importances_
    except Exception:
        # 相关性回退
        imp = X.corrwith(y).abs().fillna(0.0).values
    out = pd.DataFrame({"Feature_Name": Xcols, "Importance": imp})
    # 归一化到和为1
    s = float(out["Importance"].sum())
    if s > 0:
        out["Importance"] = out["Importance"] / s
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


# =============================
# 从特征权重提取政策权重
# =============================
def _pick_weight(imp_df: pd.DataFrame, patterns: list[str]) -> float:
    mask = imp_df["Feature_Name"].astype(str).str.lower()
    w = imp_df.loc[mask.str.contains("|".join(patterns)), "Importance"].sum()
    return float(w)


def extract_policy_weights(importance_df: pd.DataFrame) -> dict[str, float]:
    """
    解析得到政策维度权重: tariff/subsidy/export/rd/demand/cost
    若无法解析则均分
    """
    df = importance_df.copy()
    total = float(df["Importance"].sum())
    if total <= 0:
        df["Importance"] = 1.0 / max(len(df), 1)
        total = float(df["Importance"].sum())

    tariff = _pick_weight(df, ["tariff", "duty", "关税"])
    subsidy = _pick_weight(df, ["subsidy", "grant", "补贴"])
    exportc = _pick_weight(df, ["export control", "sanction", "管制", "禁运", "限制"])
    rd = _pick_weight(df, ["r&d", "research", "研发"])
    demand = _pick_weight(df, ["demand", "market", "自给", "需求"])
    cost = _pick_weight(df, ["cost", "price", "成本"])

    base = total if total > 0 else 1.0
    w = {
        "tariff": tariff / base,
        "subsidy": subsidy / base,
        "export": exportc / base,
        "rd": rd / base,
        "demand": demand / base,
        "cost": cost / base,
    }
    if sum(w.values()) == 0:
        w = {k: 1.0 / 6.0 for k in w}
    return w


# =============================
# 收益矩阵与纳什求解
# =============================
def _domain_shares(domain: str) -> tuple[float, float]:
    d = domain.lower()
    if d == "low":
        return 0.7, 0.3
    if d == "high":
        return 0.3, 0.7
    return 0.5, 0.5

def build_payoff_matrices_2d(w: dict[str, float], domain: str) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    us_strats = ["Restrict", "Subsidize", "Cooperate"]
    cn_strats = ["Buy", "Counter", "Partial"]
    A = np.zeros((3, 3))

    t_w = float(w.get("tariff", 0.0))
    s_w = float(w.get("subsidy", 0.0))
    econ_share, sec_share = _domain_shares(domain)

    def us_base(action: str) -> tuple[float, float, float, float]:
        if action == "Restrict":
            base_e = 9.0 if econ_share > 0.6 else (6.0 if abs(econ_share - 0.5) < 1e-12 else 4.0)
            base_s = 4.0 if sec_share > 0.6 else (6.0 if abs(sec_share - 0.5) < 1e-12 else 8.0)
            econ_w, sec_w = t_w, t_w * 0.8
        elif action == "Subsidize":
            base_e = 4.0 if econ_share > 0.6 else (6.0 if abs(econ_share - 0.5) < 1e-12 else 9.0)
            base_s = 9.0 if sec_share > 0.6 else (6.0 if abs(sec_share - 0.5) < 1e-12 else 4.0)
            econ_w, sec_w = s_w * 0.8, s_w
        else:
            base_e, base_s = 7.0, 7.0
            avg = (t_w + s_w) / 2.0
            econ_w, sec_w = avg, avg
        return base_e, base_s, econ_w, sec_w

    def cn_adjust(action: str) -> float:
        if action == "Counter":
            return 0.6
        if action == "Buy":
            return 1.3
        return 0.9

    for i, us_a in enumerate(us_strats):
        base_e, base_s, ew, sw = us_base(us_a)
        for j, cn_a in enumerate(cn_strats):
            adj = cn_adjust(cn_a)
            us_payoff = base_e * adj * ew * econ_share + base_s * adj * sw * sec_share
            A[i, j] = -us_payoff

    B = -A.copy()
    return us_strats, cn_strats, A, B
def build_payoff_matrices(w: dict[str, float]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    构建 3×3 策略收益矩阵
    策略: Restrict, Subsidize, Cooperate
    返回 strategies, A(China), B(US)
    """
    strategies = ["Restrict", "Subsidize", "Cooperate"]
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))

    def payoff(action_self: str, action_other: str) -> float:
        t, s, e, r, d, c = w["tariff"], w["subsidy"], w["export"], w["rd"], w["demand"], w["cost"]
        if action_self == "Restrict" and action_other == "Restrict":
            val = -(t + e) - 0.3 * c
        elif action_self == "Restrict" and action_other == "Subsidize":
            val = 0.5 * e - 0.3 * s - 0.2 * c
        elif action_self == "Restrict" and action_other == "Cooperate":
            val = 0.2 * e - 0.6 * c
        elif action_self == "Subsidize" and action_other == "Restrict":
            val = -0.3 * e + 0.6 * s - 0.4 * c
        elif action_self == "Subsidize" and action_other == "Subsidize":
            val = 0.4 * s - 0.2 * c
        elif action_self == "Subsidize" and action_other == "Cooperate":
            val = 0.5 * s + 0.3 * r - 0.2 * c
        elif action_self == "Cooperate" and action_other == "Restrict":
            val = -0.5 * e - 0.2 * c
        elif action_self == "Cooperate" and action_other == "Subsidize":
            val = 0.3 * r + 0.2 * d - 0.1 * c
        elif action_self == "Cooperate" and action_other == "Cooperate":
            val = 0.6 * r + 0.6 * d - 0.1 * c
        else:
            val = 0.0
        return float(val)

    for i, a_c in enumerate(strategies):
        for j, a_u in enumerate(strategies):
            A[i, j] = payoff(a_c, a_u)
    B = -A.copy()
    return strategies, A, B


def find_pure_saddle(A: np.ndarray) -> list[tuple[int, int]]:
    eq: list[tuple[int, int]] = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] >= A[:, j].max() - 1e-12 and A[i, j] <= A[i, :].min() - 1e-12:
                eq.append((i, j))
            elif A[i, j] >= A[:, j].max() - 1e-12 and A[i, j] <= A[i, :].min() + 1e-12:
                eq.append((i, j))
    return eq


def solve_zero_sum_equilibrium(A: np.ndarray) -> dict:
    results = {"pure": [], "mixed": []}
    results["pure"] = find_pure_saddle(A)
    try:
        import nashpy as nash
        game = nash.Game(A, -A)
        try:
            p_row, p_col = game.solve_zero_sum()
            results["mixed"] = [(p_row, p_col)]
        except Exception:
            results["mixed"] = list(game.support_enumeration())
    except Exception:
        results["mixed"] = []
    return results


# =============================
# 可视化与保存
# =============================
def plot_payoff_heatmap(A: np.ndarray, row_labels: list[str], col_labels: list[str], title: str):
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(A, cmap="viridis")
    plt.colorbar(label="Payoff")
    plt.xticks(range(len(col_labels)), col_labels)
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            color = "white" if A[i, j] < 0 else "black"
            plt.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", color=color)
    plt.tight_layout()
    plt.show()


def adjust_domain_weights(base: dict[str, float], domain: str) -> dict[str, float]:
    """
    根据芯片档次调整政策权重
    - low: 成熟制程/通用品 → 关税/成本/需求更重要
    - mid: 均衡
    - high: 先进制程/AI/HPC → 出口管制/R&D更重要, 成本权重弱化
    """
    domain = domain.lower()
    if domain == "low":
        mult = {"tariff": 1.2, "subsidy": 0.8, "export": 0.8, "rd": 0.6, "demand": 1.1, "cost": 1.3}
    elif domain == "high":
        mult = {"tariff": 0.8, "subsidy": 1.2, "export": 1.4, "rd": 1.5, "demand": 1.0, "cost": 0.7}
    else:  # mid
        mult = {"tariff": 1.0, "subsidy": 1.0, "export": 1.0, "rd": 1.0, "demand": 1.0, "cost": 1.0}

    w = {k: base.get(k, 0.0) * mult[k] for k in mult}
    s = sum(w.values())
    if s > 0:
        w = {k: v / s for k, v in w.items()}
    return w


def save_results_json(out_path: Path, strategies: list[str], A: np.ndarray, B: np.ndarray, results: dict, weights: dict):
    out = {
        "strategies": strategies,
        "A": A.tolist(),
        "B": B.tolist(),
        "pure": results.get("pure", []),
        "mixed": [
            {
                "china": [float(x) for x in eq[0]],
                "us": [float(y) for y in eq[1]],
            }
            for eq in results.get("mixed", [])
        ],
        "policy_weights": {k: float(v) for k, v in weights.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# =============================
# 运行主流程
# =============================
def run(
    data_path: Path | str | None = None,
    target_hint: str | None = None,
    save_json: bool = True,
    no_plot: bool = False,
) -> dict:
    df_raw = load_raw(data_path)
    df_num = clean_numeric(df_raw)
    target_col = pick_target_column(df_num, target_hint=target_hint)
    importance_df = compute_feature_importance(df_num, target_col=target_col)
    base_weights = extract_policy_weights(importance_df)

    # 单域：基准结果
    row_mid, col_mid, A_mid, B_mid = build_payoff_matrices_2d(base_weights, "mid")
    results = solve_zero_sum_equilibrium(A_mid)

    # 三域：低/中/高端新品
    domains = ["low", "mid", "high"]
    domain_outputs = {}
    repo_root = Path(__file__).resolve().parents[1]
    for d in domains:
        row_d, col_d, A_d, B_d = build_payoff_matrices_2d(base_weights, d)
        r_d = solve_zero_sum_equilibrium(A_d)
        domain_outputs[d] = {"weights": base_weights, "row": row_d, "col": col_d, "A": A_d, "B": B_d, "results": r_d}
        if not no_plot:
            title_map = {"low": "China payoff (Low-end)", "mid": "China payoff (Mid-end)", "high": "China payoff (High-end)"}
            plot_payoff_heatmap(A_d, row_d, col_d, title_map[d])
        if save_json:
            out_json_d = repo_root / "data" / "processed" / f"Q3_game_theory_{d}.json"
            save_results_json(out_json_d, row_d, A_d, B_d, r_d, base_weights)

    # 保存基准总览
    if save_json:
        out_json_base = repo_root / "data" / "processed" / "Q3_game_theory_results_base.json"
        save_results_json(out_json_base, row_mid, A_mid, B_mid, results, base_weights)

    return {
        "target_col": target_col,
        "importance_df": importance_df,
        "policy_weights": base_weights,
        "strategies": row_mid,
        "A": A_mid,
        "B": B_mid,
        "results": results,
        "domains": domain_outputs,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="数据文件路径, 默认为Q3英文列CSV")
    parser.add_argument("--target_hint", type=str, default=None, help="目标列关键词提示, 如 'growth' 或 '自给'")
    parser.add_argument("--no_plot", action="store_true", help="不绘图")
    parser.add_argument("--no_save", action="store_true", help="不保存JSON结果")
    args = parser.parse_args()

    run(
        data_path=args.data,
        target_hint=args.target_hint,
        save_json=not args.no_save,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()