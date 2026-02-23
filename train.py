"""
========================================================
  FILE 3: train.py
  Train CatBoost model for Sri Lanka House Price Prediction

  Input:  data/houses_clean_new_one.csv
  Output: models/catboost_model.cbm
          outputs/plots/
========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os, warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lime.lime_tabular import LimeTabularExplainer

os.makedirs("models",        exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# ════════════════════════════════════════════════════════
#  1. LOAD DATA
# ════════════════════════════════════════════════════════
df = pd.read_csv("data/houses_clean_new_one.csv")
print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}\n")

FEATURES = ["bedrooms", "bathrooms", "land_perches", "floor_sqft", "district", "suburb"]
TARGET   = "price_log"
CAT_COLS = ["district", "suburb"]
CAT_IDX  = [FEATURES.index(c) for c in CAT_COLS]

X = df[FEATURES].copy()
y = df[TARGET].copy()

for col in CAT_COLS:
    X[col] = X[col].fillna("Unknown").astype(str)

# ════════════════════════════════════════════════════════
#  2. TRAIN / VALIDATION / TEST SPLIT  (70 / 15 / 15)
# ════════════════════════════════════════════════════════
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

print(f"Train : {len(X_train)} rows (70%)")
print(f"Val   : {len(X_val)}  rows (15%)")
print(f"Test  : {len(X_test)}  rows (15%)\n")

train_pool = Pool(X_train, y_train, cat_features=CAT_IDX)
val_pool   = Pool(X_val,   y_val,   cat_features=CAT_IDX)
test_pool  = Pool(X_test,  y_test,  cat_features=CAT_IDX)

# ════════════════════════════════════════════════════════
#  3. TRAIN CATBOOST
# ════════════════════════════════════════════════════════
print("=" * 55)
print("  Training CatBoost Regressor...")
print("=" * 55)

model = CatBoostRegressor(
    iterations            = 1000,
    learning_rate         = 0.05,
    depth                 = 6,
    l2_leaf_reg           = 3,
    loss_function         = "RMSE",
    eval_metric           = "RMSE",
    random_seed           = 42,
    verbose               = 100,
    early_stopping_rounds = 50,
)
model.fit(train_pool, eval_set=val_pool, plot=False)
model.save_model("models/catboost_model.cbm")
print(f"\n  Best iteration: {model.best_iteration_}")
print(f"  Model saved -> models/catboost_model.cbm\n")

# ════════════════════════════════════════════════════════
#  4. EVALUATE
# ════════════════════════════════════════════════════════
def get_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "R2"       : round(r2_score(y_true_log, y_pred_log), 4),
        "RMSE_M"   : round(np.sqrt(mean_squared_error(y_true, y_pred)) / 1e6, 2),
        "MAE_M"    : round(mean_absolute_error(y_true, y_pred) / 1e6, 2),
        "MAPE_pct" : round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2),
    }

train_metrics = get_metrics(y_train.values, model.predict(train_pool))
val_metrics   = get_metrics(y_val.values,   model.predict(val_pool))
test_metrics  = get_metrics(y_test.values,  model.predict(test_pool))

print("=" * 55)
print(f"  {'Metric':<12} {'Train':>8} {'Val':>8} {'Test':>8}")
print("  " + "-" * 40)
for k in ["R2", "RMSE_M", "MAE_M", "MAPE_pct"]:
    print(f"  {k:<12} {train_metrics[k]:>8} {val_metrics[k]:>8} {test_metrics[k]:>8}")
print("=" * 55)

# ════════════════════════════════════════════════════════
#  5. PLOTS
# ════════════════════════════════════════════════════════

# -- Plot 1: Actual vs Predicted --
y_pred_test = np.expm1(model.predict(test_pool))
y_true_test = np.expm1(y_test.values)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_true_test/1e6, y_pred_test/1e6, alpha=0.4, color="#2563EB", s=25, edgecolors="none")
mx = max(y_true_test.max(), y_pred_test.max()) / 1e6
ax.plot([0, mx], [0, mx], "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Price (LKR Millions)", fontsize=12)
ax.set_ylabel("Predicted Price (LKR Millions)", fontsize=12)
ax.set_title(f"Actual vs Predicted — Test Set\nR² = {test_metrics['R2']}", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/plots/actual_vs_predicted.png", dpi=150)
plt.close()
print("  Saved: actual_vs_predicted.png")

# -- Plot 2: Residuals --
residuals = y_pred_test - y_true_test
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_pred_test/1e6, residuals/1e6, alpha=0.4, color="#7C3AED", s=25, edgecolors="none")
ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel("Predicted Price (LKR Millions)", fontsize=12)
ax.set_ylabel("Residual (Predicted − Actual, Millions)", fontsize=12)
ax.set_title("Residual Plot — Test Set", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/plots/residuals.png", dpi=150)
plt.close()
print("  Saved: residuals.png")

# -- Plot 3: Learning Curve --
evals = model.get_evals_result()
train_rmse = evals.get("learn", {}).get("RMSE", [])
val_rmse   = evals.get("validation", {}).get("RMSE", [])
if train_rmse and val_rmse:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_rmse, label="Train RMSE", color="#2563EB", linewidth=1.5)
    ax.plot(val_rmse,   label="Val RMSE",   color="#DC2626", linewidth=1.5)
    ax.axvline(model.best_iteration_, color="green", linestyle="--",
               label=f"Best iteration: {model.best_iteration_}")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("CatBoost Learning Curve", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/learning_curve.png", dpi=150)
    plt.close()
    print("  Saved: learning_curve.png")

# -- Plot 4: CatBoost Feature Importance --
fi = pd.Series(model.get_feature_importance(), index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
fi.plot(kind="barh", ax=ax, color="#2563EB")
ax.set_xlabel("Feature Importance (%)", fontsize=12)
ax.set_title("CatBoost Feature Importance", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/plots/feature_importance.png", dpi=150)
plt.close()
print("  Saved: feature_importance.png")

# -- Plot 5: Price Distribution --
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["price_lkr"]/1e6, bins=40, color="#2563EB", edgecolor="white")
axes[0].set_xlabel("Price (LKR Millions)")
axes[0].set_title("Price Distribution (Raw)")
axes[1].hist(df["price_log"], bins=40, color="#7C3AED", edgecolor="white")
axes[1].set_xlabel("Log Price")
axes[1].set_title("Price Distribution (Log-transformed)")
plt.tight_layout()
plt.savefig("outputs/plots/price_distribution.png", dpi=150)
plt.close()
print("  Saved: price_distribution.png")

# ════════════════════════════════════════════════════════
#  6. SHAP EXPLAINABILITY
# ════════════════════════════════════════════════════════
print("\n  Computing SHAP values...")

sample_size = min(300, len(X_test))
X_sample = X_test.sample(sample_size, random_state=42).reset_index(drop=True)
sample_pool = Pool(X_sample, cat_features=CAT_IDX)

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_pool)
shap_df     = pd.DataFrame(shap_values, columns=FEATURES)

# SHAP Summary (beeswarm)
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, feature_names=FEATURES, show=False)
plt.title("SHAP Summary — Feature Impact on House Price", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig("outputs/plots/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_summary.png")

# SHAP Bar (mean absolute)
mean_shap = np.abs(shap_df).mean().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
mean_shap.plot(kind="barh", ax=ax, color="#2563EB")
ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
ax.set_title("Feature Importance via SHAP", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/plots/shap_bar.png", dpi=150)
plt.close()
print("  Saved: shap_bar.png")

# SHAP Waterfall for 3 examples (cheap, median, expensive)
price_lkr_sample = df.loc[X_sample.index, "price_lkr"] if len(df) > max(X_sample.index) else df["price_lkr"].iloc[X_sample.index]
examples = {
    "Cheapest"      : np.argmin(np.expm1(y_test.values)),
    "Median"        : np.argmin(np.abs(np.expm1(y_test.values) - np.median(np.expm1(y_test.values)))),
    "Most Expensive": np.argmax(np.expm1(y_test.values)),
}
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

for label, pos in examples.items():
    if pos >= len(X_test_reset):
        continue
    x_row  = X_test_reset.iloc[[pos]]
    sv     = explainer.shap_values(Pool(x_row, cat_features=CAT_IDX))
    expl   = shap.Explanation(
        values        = sv[0],
        base_values   = explainer.expected_value,
        data          = x_row.iloc[0].values,
        feature_names = FEATURES,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.waterfall_plot(expl, show=False, max_display=10)
    actual_price = np.expm1(y_test_reset.iloc[pos]) / 1e6
    plt.title(f"SHAP Waterfall — {label} House (Actual: LKR {actual_price:.1f}M)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"outputs/plots/shap_waterfall_{label.lower().replace(' ','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: shap_waterfall_{label.lower().replace(' ','_')}.png")

# ════════════════════════════════════════════════════════
#  7. LIME EXPLAINABILITY
# ════════════════════════════════════════════════════════
print("\n  Computing LIME explanation...")

# Encode categoricals for LIME (needs numeric input)
from sklearn.preprocessing import LabelEncoder
X_encoded = X.copy()
le_d = LabelEncoder().fit(X["district"])
le_s = LabelEncoder().fit(X["suburb"])
X_encoded["district"] = le_d.transform(X["district"])
X_encoded["suburb"]   = le_s.transform(X["suburb"])

X_train_enc = X_encoded.loc[X_train.index].values
X_test_enc  = X_encoded.loc[X_test.index].values

def predict_fn(X_arr):
    """Convert encoded numpy array back for CatBoost prediction."""
    df_temp = pd.DataFrame(X_arr, columns=FEATURES)
    df_temp["district"] = le_d.inverse_transform(df_temp["district"].astype(int))
    df_temp["suburb"]   = le_s.inverse_transform(df_temp["suburb"].astype(int))
    pool = Pool(df_temp, cat_features=CAT_IDX)
    return model.predict(pool)

lime_explainer = LimeTabularExplainer(
    training_data      = X_train_enc,
    feature_names      = FEATURES,
    mode               = "regression",
    discretize_continuous = True,
    random_state       = 42,
)

# Explain one test instance (median priced)
mid_pos = np.argmin(np.abs(np.expm1(y_test.values) - np.median(np.expm1(y_test.values))))
lime_exp = lime_explainer.explain_instance(
    data_row  = X_test_enc[mid_pos],
    predict_fn = predict_fn,
    num_features = 6,
)

fig = lime_exp.as_pyplot_figure()
actual_price = np.expm1(y_test.values[mid_pos]) / 1e6
plt.title(f"LIME Explanation — Median House (Actual: LKR {actual_price:.1f}M)", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/plots/lime_explanation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: lime_explanation.png")

# ════════════════════════════════════════════════════════
#  8. PARTIAL DEPENDENCE PLOTS (PDP)
# ════════════════════════════════════════════════════════
print("\n  Computing Partial Dependence Plots...")

numeric_features = ["bedrooms", "bathrooms", "land_perches", "floor_sqft"]

for feat in numeric_features:
    feat_vals  = np.linspace(X_train[feat].min(), X_train[feat].max(), 40)
    X_pdp_base = X_train.copy().reset_index(drop=True)
    pdp_preds  = []

    for val in feat_vals:
        X_pdp_temp = X_pdp_base.copy()
        X_pdp_temp[feat] = val
        pool_pdp = Pool(X_pdp_temp, cat_features=CAT_IDX)
        avg_pred = np.expm1(model.predict(pool_pdp)).mean() / 1e6
        pdp_preds.append(avg_pred)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(feat_vals, pdp_preds, color="#2563EB", linewidth=2)
    ax.set_xlabel(feat.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Average Predicted Price (LKR M)", fontsize=12)
    ax.set_title(f"Partial Dependence Plot: {feat.replace('_',' ').title()}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"outputs/plots/pdp_{feat}.png", dpi=150)
    plt.close()
    print(f"  Saved: pdp_{feat}.png")

# ════════════════════════════════════════════════════════
#  DONE
# ════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"  ALL DONE!")
print(f"\n  Final Test Results:")
print(f"    R²      : {test_metrics['R2']}")
print(f"    RMSE    : LKR {test_metrics['RMSE_M']}M")
print(f"    MAE     : LKR {test_metrics['MAE_M']}M")
print(f"    MAPE    : {test_metrics['MAPE_pct']}%")
print(f"\n  Plots saved in: outputs/plots/")
print(f"  Model saved  : models/catboost_model.cbm")
print(f"{'='*55}\n")