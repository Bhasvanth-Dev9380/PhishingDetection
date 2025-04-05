from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

# Load models
scaler = joblib.load("scaler.pkl")
iso_forest = joblib.load("iso_forest_model.pkl")
xgb_model = joblib.load("xgb_hybrid_model.pkl")
trained_columns = joblib.load("trained_columns.pkl")

app = Flask(__name__)
CORS(app)

@app.route('/detect', methods=['POST'])
def detect_phishing():
    try:
        data = request.get_json()
        email = data.get("email", "unknown")
        features = data.get("features")

        print(f"\n📥 Request received from: {email}")
        print("📊 Raw Features:", features)

        if not features:
            print("❌ No features provided.")
            return jsonify({"error": "No features provided"}), 400

        # Convert to DataFrame
        df_input = pd.DataFrame([features])

        # Align columns
        for col in trained_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[trained_columns]

        print("✅ Aligned Input:", df_input.to_dict(orient='records')[0])

        # Scale the features
        df_scaled = scaler.transform(df_input)

        # Isolation Forest anomaly prediction
        anomaly_score = iso_forest.predict(df_scaled)[0]
        is_anomaly = 1 if anomaly_score == -1 else 0
        print(f"🔍 Anomaly Score: {anomaly_score} → Anomaly Detected: {bool(is_anomaly)}")

        # Add anomaly score as a feature
        df_enhanced = pd.DataFrame(df_scaled, columns=trained_columns)
        df_enhanced["is_anomaly"] = is_anomaly

        # XGBoost prediction
        prediction = int(xgb_model.predict(df_enhanced)[0])
        confidence = float(xgb_model.predict_proba(df_enhanced)[0][prediction])
        print(f"🎯 Prediction: {'PHISHING' if prediction else 'LEGITIMATE'} | Confidence: {round(confidence, 4)}")

        return jsonify({
            "email": email,
            "phishing": bool(prediction),
            "confidence": round(confidence, 4),
            "anomaly_detected": bool(is_anomaly)
        })

    except Exception as e:
        print("🚨 Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/')
def health_check():
    return "Phishing Detection API is running."

if __name__ == '__main__':
    app.run(debug=True)
