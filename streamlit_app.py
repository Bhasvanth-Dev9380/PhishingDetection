import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# === Load saved models ===
scaler = joblib.load("scaler.pkl")
iso_forest = joblib.load("iso_forest_model.pkl")
xgb_model = joblib.load("xgb_hybrid_model.pkl")
trained_columns = joblib.load("trained_columns.pkl")

# === Load dataset for sample defaults ===
df = pd.read_csv("dataset_phishing.csv")
sample_df = df.drop(columns=["url", "status"], errors='ignore')
sample_df = sample_df.loc[:, (sample_df != sample_df.iloc[0]).any()]
default_samples = sample_df.head(10).reset_index(drop=True)

# App config
st.set_page_config(page_title="Phishing Detection", layout="centered")
st.title("🛡️ Phishing Detection Dashboard (Hybrid Model)")

# === Sidebar Switch ===
mode = st.sidebar.radio("Choose Mode:", ["Manual Input", "CSV Upload"])

if mode == "Manual Input":
    st.markdown("Select a sample row or enter features manually to predict whether it's **phishing** or **legitimate**.")

    # Sample row selection
    selected_sample = st.selectbox("📋 Choose a sample input (from dataset)", options=list(range(1, 11)), index=0)
    selected_data = default_samples.iloc[selected_sample - 1]

    # Manual input
    input_data = {}
    cols = st.columns(2)
    for i, col in enumerate(trained_columns):
        default_val = selected_data[col] if col in selected_data else 0.0
        input_data[col] = cols[i % 2].number_input(f"{col}", value=float(default_val))

    if st.button("🔍 Detect Phishing"):

        df_input = pd.DataFrame([input_data])

        for col in trained_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[trained_columns]

        df_scaled = scaler.transform(df_input)
        anomaly_score = iso_forest.predict(df_scaled)[0]
        is_anomaly = 1 if anomaly_score == -1 else 0

        df_enhanced = pd.DataFrame(df_scaled, columns=trained_columns)
        df_enhanced["is_anomaly"] = is_anomaly

        prediction = xgb_model.predict(df_enhanced)[0]
        confidence = xgb_model.predict_proba(df_enhanced)[0]

        label = "PHISHING" if prediction == 1 else "LEGITIMATE"
        st.subheader(f"🛡️ Prediction: {label}")
        st.write(f"🔍 Anomaly Detected: {'Yes' if is_anomaly else 'No'}")
        st.write(f"📊 Confidence: Legitimate = {confidence[0]:.2f}, Phishing = {confidence[1]:.2f}")

elif mode == "CSV Upload":
    st.markdown("📁 Upload a CSV file with the **same feature columns** as the dataset (excluding 'url' and 'status').")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        csv_df = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview:")
        st.dataframe(csv_df.head())

        # Align columns
        for col in trained_columns:
            if col not in csv_df.columns:
                csv_df[col] = 0
        csv_df = csv_df[trained_columns]

        # Scale + Isolation Forest
        scaled_data = scaler.transform(csv_df)
        anomaly_scores = iso_forest.predict(scaled_data)
        anomalies = pd.Series(anomaly_scores).map(lambda x: 1 if x == -1 else 0)

        # Add anomaly column
        enhanced_df = pd.DataFrame(scaled_data, columns=trained_columns)
        enhanced_df["is_anomaly"] = anomalies

        # XGBoost predictions
        preds = xgb_model.predict(enhanced_df)
        confs = xgb_model.predict_proba(enhanced_df)

        # Create results DataFrame
        result_df = csv_df.copy()
        result_df["Anomaly"] = anomalies
        result_df["Prediction"] = ["PHISHING" if p == 1 else "LEGITIMATE" for p in preds]
        result_df["Confidence_Phishing"] = confs[:, 1]
        result_df["Confidence_Legitimate"] = confs[:, 0]

        st.success("✅ Predictions completed.")

        # === Show pie chart ===
        st.subheader("📊 Prediction Summary")
        chart_data = result_df["Prediction"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(chart_data, labels=chart_data.index, autopct='%1.1f%%', colors=['#ff9999', '#99ff99'])
        ax.axis("equal")
        st.pyplot(fig)

        # === Flag IP/DNS
        if 'ip' in result_df.columns:
            flagged_ips = result_df[result_df['ip'] == 1]
            if not flagged_ips.empty:
                st.warning(f"⚠️ {len(flagged_ips)} entries use direct IPs — suspicious!")
        if 'dns_record' in result_df.columns:
            missing_dns = result_df[result_df['dns_record'] == 0]
            if not missing_dns.empty:
                st.warning(f"⚠️ {len(missing_dns)} entries have missing DNS records.")

        # === Filter phishing
        st.subheader("🔎 View Filtered Results")
        filter_option = st.radio("Filter:", ["All", "Only PHISHING", "Only LEGITIMATE"], horizontal=True)

        filtered_df = result_df.copy()
        if filter_option == "Only PHISHING":
            filtered_df = result_df[result_df["Prediction"] == "PHISHING"]
        elif filter_option == "Only LEGITIMATE":
            filtered_df = result_df[result_df["Prediction"] == "LEGITIMATE"]

        # === Paginate table display
        st.subheader("📋 Paginated Results")
        rows_per_page = st.selectbox("Rows per page:", [10, 25, 50, 100], index=1)
        total_rows = len(filtered_df)
        total_pages = (total_rows - 1) // rows_per_page + 1
        page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
        start = (page - 1) * rows_per_page
        end = start + rows_per_page

        st.dataframe(filtered_df.iloc[start:end])

        # === Download options
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        st.subheader("⬇️ Download Reports")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download All Predictions", convert_df(result_df), "phishing_predictions.csv", "text/csv")
        with col2:
            phishing_only = result_df[result_df["Prediction"] == "PHISHING"]
            st.download_button("📥 Download Only PHISHING", convert_df(phishing_only), "phishing_only.csv", "text/csv")
