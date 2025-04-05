import pandas as pd
import joblib

# === Step 1: Load the saved models ===
scaler = joblib.load("scaler.pkl")
iso_forest = joblib.load("iso_forest_model.pkl")
xgb_model = joblib.load("xgb_hybrid_model.pkl")

# === Step 2: Define input data (must match trained columns) ===
# You can replace these values or load from a file
input_data = {
    'length_url': 60,
    'length_hostname': 20,
    'ip': 0,
    'nb_dots': 2,
    'nb_hyphens': 1,
    'nb_at': 0,
    'nb_qm': 1,
    'nb_and': 0,
    'nb_or': 0,
    'domain_registration_length': 365,
    'domain_age': 800,
    'web_traffic': 100000,
    'dns_record': 1,
    'google_index': 1,
    'page_rank': 3,
    # Include ALL other features from dataset_phishing.csv
}

# === Step 3: Align features with training data ===
df_input = pd.DataFrame([input_data])
trained_columns = joblib.load("trained_columns.pkl")  # Saved during training

# Fill missing and reorder
for col in trained_columns:
    if col not in df_input.columns:
        df_input[col] = 0
df_input = df_input[trained_columns]

# === Step 4: Scale the input ===
df_scaled = scaler.transform(df_input)

# === Step 5: Add anomaly feature from Isolation Forest ===
anomaly_score = iso_forest.predict(df_scaled)[0]
is_anomaly = 1 if anomaly_score == -1 else 0
df_enhanced = pd.DataFrame(df_scaled, columns=trained_columns)
df_enhanced["is_anomaly"] = is_anomaly

# === Step 6: Run XGBoost prediction ===
prediction = xgb_model.predict(df_enhanced)[0]
confidence = xgb_model.predict_proba(df_enhanced)[0]

label = "PHISHING" if prediction == 1 else "LEGITIMATE"
print(f"\n🛡️ Prediction: {label}")
print(f"🔍 Anomaly: {'Yes' if is_anomaly else 'No'}")
print(f"📊 Confidence: Legitimate = {confidence[0]:.2f}, Phishing = {confidence[1]:.2f}")
