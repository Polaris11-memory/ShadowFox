"""
Boston Housing Price Prediction
================================
A complete regression pipeline including:
  - Data loading & exploration
  - Preprocessing (missing values, outliers, scaling)
  - Model selection (Linear Regression, Decision Tree, Gradient Boosting)
  - Training & Evaluation (MSE, RMSE, R²)
  - Fine-tuning via GridSearchCV
  - Visualization of results
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  BOSTON HOUSING PRICE PREDICTION")
print("=" * 60)

df = pd.read_csv("/mnt/user-data/uploads/HousingData.csv")

FEATURE_DESCRIPTIONS = {
    "CRIM":    "Per-capita crime rate by town",
    "ZN":      "% residential land zoned for large lots",
    "INDUS":   "% non-retail business acres per town",
    "CHAS":    "Charles River dummy variable (1=bounds river)",
    "NOX":     "Nitric oxides concentration (parts per 10M)",
    "RM":      "Avg number of rooms per dwelling",
    "AGE":     "% owner-occupied units built before 1940",
    "DIS":     "Weighted distances to employment centres",
    "RAD":     "Index of accessibility to radial highways",
    "TAX":     "Full-value property-tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B":       "1000(Bk − 0.63)² where Bk = % Black residents",
    "LSTAT":   "% lower-status population",
    "MEDV":    "Median value of owner-occupied homes ($1000s)  ← TARGET",
}

print(f"\n📦  Dataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋  Feature descriptions:")
for col, desc in FEATURE_DESCRIPTIONS.items():
    print(f"    {col:<9} {desc}")

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n\n── 2. EXPLORATORY DATA ANALYSIS ──────────────────────────")
print(f"\nMissing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nTarget (MEDV) stats:\n{df['MEDV'].describe().to_string()}")

features = [c for c in df.columns if c != "MEDV"]
X_raw = df[features]
y = df["MEDV"]

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
print("\n\n── 3. PREPROCESSING ──────────────────────────────────────")

# 3a. Impute missing values with median (robust to outliers)
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=features)
print(f"  ✔ Missing values imputed with column medians")

# 3b. Outlier detection using IQR (clip instead of drop to keep all rows)
def clip_outliers_iqr(df_in, factor=3.0):
    df_out = df_in.copy()
    for col in df_out.columns:
        Q1, Q3 = df_out[col].quantile(0.25), df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - factor * IQR, Q3 + factor * IQR
        clipped = df_out[col].clip(lo, hi)
        n = (df_out[col] != clipped).sum()
        if n:
            print(f"  ✔ {col}: {n} outlier(s) clipped")
        df_out[col] = clipped
    return df_out

X_clean = clip_outliers_iqr(X_imputed)

# 3c. Train / test split (80 / 20, stratified on price quartile)
price_q = pd.qcut(y, q=4, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42, stratify=price_q
)
print(f"\n  Train size : {len(X_train)}  |  Test size : {len(X_test)}")

# 3d. Feature scaling (StandardScaler fit only on train)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"  ✔ Features standardised (zero-mean, unit-variance)")

# ─────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION (BASELINE)
# ─────────────────────────────────────────────
print("\n\n── 4. BASELINE MODEL EVALUATION ─────────────────────────")

def evaluate(name, model, Xtr, ytr, Xte, yte, scaled=True):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    cv   = cross_val_score(model, Xtr, ytr, cv=5, scoring="r2")
    mse  = mean_squared_error(yte, pred)
    mae  = mean_absolute_error(yte, pred)
    r2   = r2_score(yte, pred)
    print(f"\n  {'─'*42}")
    print(f"  MODEL : {name}")
    print(f"    Test  MSE  : {mse:8.3f}")
    print(f"    Test  RMSE : {np.sqrt(mse):8.3f}")
    print(f"    Test  MAE  : {mae:8.3f}")
    print(f"    Test  R²   : {r2:8.4f}")
    print(f"    CV-5  R²   : {cv.mean():8.4f} ± {cv.std():.4f}")
    return {"name": name, "mse": mse, "rmse": np.sqrt(mse), "mae": mae,
            "r2": r2, "cv_r2": cv.mean(), "model": model, "pred": pred}

results = []
results.append(evaluate("Linear Regression",
    LinearRegression(), X_train_s, y_train, X_test_s, y_test))

results.append(evaluate("Ridge Regression (α=1)",
    Ridge(alpha=1.0), X_train_s, y_train, X_test_s, y_test))

results.append(evaluate("Decision Tree",
    DecisionTreeRegressor(random_state=42), X_train, y_train, X_test, y_test))

results.append(evaluate("Gradient Boosting",
    GradientBoostingRegressor(random_state=42), X_train, y_train, X_test, y_test))

# ─────────────────────────────────────────────
# 5. FINE-TUNING (GRADIENT BOOSTING via GridSearchCV)
# ─────────────────────────────────────────────
print("\n\n── 5. FINE-TUNING (GridSearchCV on GradientBoosting) ─────")
param_grid = {
    "n_estimators":   [100, 200, 300],
    "learning_rate":  [0.05, 0.1, 0.15],
    "max_depth":      [3, 4, 5],
    "subsample":      [0.8, 1.0],
}
gb_cv = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0
)
gb_cv.fit(X_train, y_train)
print(f"  Best params : {gb_cv.best_params_}")
print(f"  Best CV R²  : {gb_cv.best_score_:.4f}")

best_gb = gb_cv.best_estimator_
results.append(evaluate("GBR (fine-tuned)",
    best_gb, X_train, y_train, X_test, y_test))

# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (best model)
# ─────────────────────────────────────────────
feat_imp = pd.Series(best_gb.feature_importances_, index=features).sort_values(ascending=False)

# ─────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────
print("\n\n── 6. GENERATING VISUALISATIONS ──────────────────────────")

PALETTE = {
    "bg":      "#0f1117",
    "panel":   "#1a1d27",
    "accent1": "#7c6af7",
    "accent2": "#4fc3f7",
    "accent3": "#f48fb1",
    "accent4": "#a5d6a7",
    "text":    "#e0e0e0",
    "muted":   "#90a4ae",
    "grid":    "#2a2d3a",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["panel"],
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["muted"],
    "ytick.color":       PALETTE["muted"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "axes.titlepad":     14,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor(PALETTE["bg"])

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3,
                       top=0.94, bottom=0.04, left=0.07, right=0.97)

fig.suptitle("Boston Housing Price Prediction — Model Report",
             fontsize=20, fontweight="bold", color=PALETTE["text"], y=0.975)

# ── 7a. Correlation heatmap ──────────────────
ax1 = fig.add_subplot(gs[0, 0])
corr = pd.concat([X_clean, y], axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax1, cmap="coolwarm", center=0,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.4, linecolor=PALETTE["bg"],
            cbar_kws={"shrink": 0.8})
ax1.set_title("Feature Correlation Matrix", color=PALETTE["text"])
ax1.tick_params(colors=PALETTE["muted"], labelsize=8)

# ── 7b. Target distribution ──────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(y, bins=30, color=PALETTE["accent1"], edgecolor=PALETTE["bg"], alpha=0.85)
ax2.axvline(y.mean(), color=PALETTE["accent2"], lw=2, ls="--", label=f"Mean={y.mean():.1f}")
ax2.axvline(y.median(), color=PALETTE["accent3"], lw=2, ls=":", label=f"Median={y.median():.1f}")
ax2.set_title("MEDV (Target) Distribution")
ax2.set_xlabel("Median Home Value ($1000s)")
ax2.set_ylabel("Frequency")
ax2.legend(fontsize=9)
ax2.grid(True)

# ── 7c. Model comparison bar chart ───────────
ax3 = fig.add_subplot(gs[1, 0])
names  = [r["name"] for r in results]
r2vals = [r["r2"]   for r in results]
colors = [PALETTE["accent1"], PALETTE["accent4"], PALETTE["accent3"],
          PALETTE["accent2"], "#ffd54f"]
bars = ax3.barh(names, r2vals, color=colors[:len(names)],
                edgecolor=PALETTE["bg"], height=0.55)
for bar, val in zip(bars, r2vals):
    ax3.text(val + 0.003, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=9, color=PALETTE["text"])
ax3.axvline(0.9, color="white", lw=1, ls="--", alpha=0.4, label="R²=0.9 target")
ax3.set_xlim(0, 1.05)
ax3.set_title("Model Comparison — Test R² Score")
ax3.set_xlabel("R² Score (higher is better)")
ax3.legend(fontsize=8)
ax3.grid(True, axis="x")

# ── 7d. RMSE comparison ──────────────────────
ax4 = fig.add_subplot(gs[1, 1])
rmses = [r["rmse"] for r in results]
bars4 = ax4.barh(names, rmses, color=colors[:len(names)],
                 edgecolor=PALETTE["bg"], height=0.55)
for bar, val in zip(bars4, rmses):
    ax4.text(val + 0.05, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=9, color=PALETTE["text"])
ax4.set_title("Model Comparison — RMSE ($1000s)")
ax4.set_xlabel("RMSE (lower is better)")
ax4.grid(True, axis="x")

# ── 7e. Predicted vs Actual (best model) ─────
ax5 = fig.add_subplot(gs[2, 0])
best_pred = results[-1]["pred"]
ax5.scatter(y_test, best_pred, alpha=0.65, s=30,
            color=PALETTE["accent1"], edgecolors="none")
lims = [min(y_test.min(), best_pred.min()) - 1,
        max(y_test.max(), best_pred.max()) + 1]
ax5.plot(lims, lims, color=PALETTE["accent2"], lw=2, ls="--", label="Perfect fit")
ax5.set_xlim(lims); ax5.set_ylim(lims)
ax5.set_title("Predicted vs Actual — GBR (Fine-tuned)")
ax5.set_xlabel("Actual MEDV ($1000s)")
ax5.set_ylabel("Predicted MEDV ($1000s)")
ax5.legend(fontsize=9)
ax5.grid(True)

# ── 7f. Residual plot ────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
residuals = y_test.values - best_pred
ax6.scatter(best_pred, residuals, alpha=0.65, s=30,
            color=PALETTE["accent3"], edgecolors="none")
ax6.axhline(0, color=PALETTE["accent2"], lw=2, ls="--")
ax6.set_title("Residual Plot — GBR (Fine-tuned)")
ax6.set_xlabel("Predicted MEDV ($1000s)")
ax6.set_ylabel("Residual (Actual − Predicted)")
ax6.grid(True)

# ── 7g. Feature importance ───────────────────
ax7 = fig.add_subplot(gs[3, 0])
fi_colors = [PALETTE["accent1"] if v >= feat_imp.median() else PALETTE["muted"]
             for v in feat_imp.values]
ax7.barh(feat_imp.index[::-1], feat_imp.values[::-1],
         color=fi_colors[::-1], edgecolor=PALETTE["bg"], height=0.6)
ax7.set_title("Feature Importance — GBR (Fine-tuned)")
ax7.set_xlabel("Importance Score")
ax7.grid(True, axis="x")

# ── 7h. Metrics summary table ────────────────
ax8 = fig.add_subplot(gs[3, 1])
ax8.axis("off")
table_data = [[r["name"],
               f"{r['mse']:.2f}",
               f"{r['rmse']:.3f}",
               f"{r['mae']:.3f}",
               f"{r['r2']:.4f}",
               f"{r['cv_r2']:.4f}"] for r in results]
col_labels = ["Model", "MSE", "RMSE", "MAE", "R²", "CV-R²"]
tbl = ax8.table(cellText=table_data, colLabels=col_labels,
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.15, 2.0)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor(PALETTE["panel"] if row % 2 == 0 else PALETTE["bg"])
    cell.set_edgecolor(PALETTE["grid"])
    cell.set_text_props(color=PALETTE["text"])
    if row == 0:
        cell.set_facecolor(PALETTE["accent1"])
        cell.set_text_props(color="white", fontweight="bold")
    if row == len(results) and col in (4, 5):
        cell.set_facecolor("#1e3a2f")
ax8.set_title("Full Metrics Summary", color=PALETTE["text"], fontsize=13, pad=80)

output_path = "/mnt/user-data/outputs/boston_housing_report.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
print(f"  ✔ Saved → {output_path}")

# ─────────────────────────────────────────────
# 8. FINAL SUMMARY
# ─────────────────────────────────────────────
best = max(results, key=lambda r: r["r2"])
print("\n\n── 7. FINAL SUMMARY ──────────────────────────────────────")
print(f"\n  🏆  Best model : {best['name']}")
print(f"      MSE        : {best['mse']:.3f}")
print(f"      RMSE       : {best['rmse']:.3f}  ($1000s)")
print(f"      MAE        : {best['mae']:.3f}  ($1000s)")
print(f"      R²         : {best['r2']:.4f}  ({best['r2']*100:.2f}% variance explained)")
print(f"\n  Top-3 predictive features:")
for feat, imp in feat_imp.head(3).items():
    print(f"      {feat:<9} importance = {imp:.4f}  — {FEATURE_DESCRIPTIONS[feat]}")
print("\n" + "=" * 60)
print("  Done. Report saved to boston_housing_report.png")
print("=" * 60)
