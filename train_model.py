import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ===========================================================
# Helper: SMAPE calculation
# ===========================================================
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true) / np.maximum(denominator, 1e-8)
    return np.mean(diff) * 100

# ===========================================================
# Load features
# ===========================================================
print("📦 Loading features...")
train_text = joblib.load("features/train_text_features.pkl")
test_text = joblib.load("features/test_text_features.pkl")
train_img = joblib.load("features/train_image_features.pkl")
test_img = joblib.load("features/test_image_features.pkl")

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Combine text + image features
print("🔗 Combining text and image features...")
X_train = np.hstack([train_text, train_img])
X_test = np.hstack([test_text, test_img])
y_train = train_df["price"].values

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

print(f"✅ Combined features: Train {X_train.shape}, Test {X_test.shape}")

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

# ===========================================================
# LightGBM Parameters (tuned)
# ===========================================================
params = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 128,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "max_depth": -1,
    "verbosity": -1,
    "random_state": 42,
}

# ===========================================================
# Train LightGBM
# ===========================================================
print("🚀 Training LightGBM model...")
train_data = lgb.Dataset(X_tr, label=y_tr)
val_data = lgb.Dataset(X_val, label=y_val)

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=2000,
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(200),
        lgb.log_evaluation(200),
    ],
)

# ===========================================================
# Evaluate Validation Performance
# ===========================================================
print("📈 Evaluating model...")
val_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, val_pred)
smape_score = smape(y_val, val_pred)

print(f"✅ Validation MAE: {mae:.4f}")
print(f"📊 Validation SMAPE: {smape_score:.2f}%")

# ===========================================================
# Ensemble: second model with different seed
# ===========================================================
print("🤖 Training second LightGBM model for ensemble...")
params["random_state"] = 99
model2 = lgb.train(params, train_data, num_boost_round=model.best_iteration)
test_pred2 = model2.predict(X_test)

# ===========================================================
# Predict on Test Set
# ===========================================================
print("📦 Generating test predictions (ensemble)...")
test_pred = model.predict(X_test)
test_pred_final = (test_pred + test_pred2) / 2
test_pred_final = np.clip(test_pred_final, 0, None)

submission = pd.DataFrame({
    "sample_id": test_df["sample_id"],
    "price": test_pred_final
})
submission.to_csv("test_out.csv", index=False)
print("💾 Saved predictions to test_out.csv (ensemble model)")

# ===========================================================
# Save Model
# ===========================================================
joblib.dump(model, "model_lgbm.pkl")
print("✅ Model saved as model_lgbm.pkl")

print("\n🎯 All done! Re-run complete — now submit test_out.csv")
