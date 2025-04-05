import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("dataset_phishing.csv")

# Drop non-numeric or irrelevant columns
X = df.drop(columns=["url", "status"], errors='ignore')

# Remove columns with zero variance
X = X.loc[:, (X != X.iloc[0]).any()]

# Encode labels: phishing = 1, legitimate = 0
true_labels = df['status'].map(lambda x: 1 if x == 'phishing' else 0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 1: Isolation Forest (unsupervised anomaly detection) ===
phishing_ratio = (true_labels == 1).sum() / len(true_labels)

iso_forest = IsolationForest(
    n_estimators=500,
    contamination=phishing_ratio,
    random_state=42
)
iso_forest.fit(X_scaled)

# Generate anomaly feature
anomaly_scores = iso_forest.predict(X_scaled)  # -1 = anomaly
anomaly_feature = pd.Series(anomaly_scores).map(lambda x: 1 if x == -1 else 0)

# === Step 2: Add anomaly feature to data ===
X_enhanced = pd.DataFrame(X_scaled, columns=X.columns)
X_enhanced["is_anomaly"] = anomaly_feature.values

# === Step 3: Train XGBoost Classifier (supervised) ===
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_enhanced, true_labels)

# === Step 4: Evaluate on full dataset ===
y_pred = xgb_model.predict(X_enhanced)

print("\n📈 Classification Report (Hybrid Model):")
print(classification_report(true_labels, y_pred, target_names=["legitimate", "phishing"]))

# === Step 5: Save everything ===
joblib.dump(xgb_model, "xgb_hybrid_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(iso_forest, "iso_forest_model.pkl")
joblib.dump(list(X.columns), "trained_columns.pkl")


print("\n✅ Models saved:")
