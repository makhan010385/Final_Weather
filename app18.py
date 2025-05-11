import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.title("🌱 Soybean Disease Severity Predictor (Weather & Variety Based)")

# Upload section
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Convert potential numeric columns from object
    for col in ['JS 95-60', 'JS93-05', 'PK -472']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter valid varieties
    ignored_cols = ['Year', 'SMW', 'Crop_Growth_Week', 'Location', 'Longitude', 'Latitude',
                    'Max_Temp', 'Min_Temp', 'Max_Humidity', 'Min_Humidity', 'No of Rainy days',
                    'Rainfall', 'Wind_Velocity', 'Disease', 'Gaurav', ' JS 90-41']
    varieties = [col for col in df.columns if col not in ignored_cols]

    selected_variety = st.selectbox("🌿 Select soybean variety", varieties)

    # 🌱 User picks sowing date
    st.subheader("📆 Select Sowing Date and Current Date")
    sowing_date = st.date_input(
        "🌱 Select Sowing Date",
        min_value=date(2021, 6, 1),
        max_value=date(2021, 12, 31),
        value=date(2021, 6, 15)
    )

    current_date = st.date_input(
        "📅 Select Date for Prediction",
        min_value=sowing_date,
        max_value=date(2021, 12, 31),
        value=sowing_date + pd.Timedelta(days=21)
    )

    # Calculate SMW and Crop Growth Week
    smw = current_date.isocalendar()[1]
    crop_growth_week = max(1, ((current_date - sowing_date).days // 7) + 1)

    st.write(f"📅 Selected Date: {current_date.strftime('%Y-%m-%d')}")
    st.write(f"📈 Standard Meteorological Week (SMW): **{smw}**")
    st.write(f"🌱 Crop Growth Week: **{crop_growth_week}** (since sowing date)")

    # Prepare data for modeling
    features = ['Max_Temp', 'Min_Temp', 'Rainfall', 'Max_Humidity',
                'Min_Humidity', 'No of Rainy days', 'Crop_Growth_Week']
    df['Crop_Growth_Week'] = pd.to_numeric(df['Crop_Growth_Week'], errors='coerce')
    df_model = df[features + [selected_variety]].dropna()

    if df_model.empty:
        st.warning("⚠️ Not enough complete data to train model for this variety.")
    else:
        st.subheader("📊 Correlation Heatmap")
        corr = df_model.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax)
        st.pyplot(fig)

        # Train model
        X = df_model[features]
        y = df_model[selected_variety]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)

        st.subheader(f"📈 Regression Results for {selected_variety}")
        st.write(f"**R² Score:** {score:.3f}")
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_
        })
        st.dataframe(coef_df)

        # Rule-based function for disease severity
        def get_disease_and_severity(PDI, max_temp, min_temp, max_hum, min_hum, rain, rainy_days, crop_week):
            if PDI > 40:
                if max_hum > 85 and min_hum > 60 and rainy_days > 2:
                    if crop_week >= 6:
                        return "Rust", "High"
                    else:
                        return "Anthracnose", "High"
                elif max_temp > 32 and min_temp > 24:
                    return "Cercospora Leaf Spot", "High"
                else:
                    return "Frogeye Leaf Spot", "High"
            elif 20 < PDI <= 40:
                if max_hum > 75 and rain > 20:
                    return "Anthracnose", "Medium"
                elif max_temp > 30 and crop_week >= 5:
                    return "Bacterial Blight", "Medium"
                else:
                    return "Frogeye Leaf Spot", "Medium"
            elif 10 < PDI <= 20:
                if rain > 10 and min_hum > 60:
                    return "Downy Mildew", "Low"
                else:
                    return "Bacterial Pustule", "Low"
            else:
                return "No Major Disease", "Safe"

        st.subheader(f"🔍 Predict Severity for {selected_variety}")
        input_data = {}
        for feature in features[:-1]:  # All except CGW
            input_data[feature] = st.number_input(f"{feature}", value=float(df[feature].mean()))
        input_data['Crop_Growth_Week'] = crop_growth_week

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            predicted_pdi = model.predict(input_df)[0]
            predicted_pdi = round(predicted_pdi, 2)

            # Apply intelligent severity rules
            disease_name, severity_level = get_disease_and_severity(
                predicted_pdi,
                input_data["Max_Temp"],
                input_data["Min_Temp"],
                input_data["Max_Humidity"],
                input_data["Min_Humidity"],
                input_data["Rainfall"],
                input_data["No of Rainy days"],
                crop_growth_week
            )

            st.success(f"✅ Predicted disease severity for **{selected_variety}**: **{predicted_pdi:.2f}% - {disease_name} ({severity_level})**")

        # 5-Day Forecast Table
        st.subheader(f"🔮 5-Day Disease Forecast for {selected_variety}")
        input_weather = {}
        for feature in features[:-1]:
            input_weather[feature] = st.number_input(
                f"Average {feature} for next 5 days", value=float(df[feature].mean())
            )

        if st.button("Predict Next 5 Days"):
            results = []
            for i in range(5):
                forecast_date = current_date + timedelta(days=i)
                crop_age_days = (forecast_date - sowing_date).days
                growth_week = max(1, crop_age_days // 7 + 1)

                input_data = input_weather.copy()
                input_data['Crop_Growth_Week'] = growth_week
                input_df = pd.DataFrame([input_data])
                predicted_pdi = model.predict(input_df)[0]
                predicted_pdi = round(predicted_pdi, 2)

                # Apply intelligent severity rules
                disease_name, severity_level = get_disease_and_severity(
                    predicted_pdi,
                    input_weather["Max_Temp"],
                    input_weather["Min_Temp"],
                    input_weather["Max_Humidity"],
                    input_weather["Min_Humidity"],
                    input_weather["Rainfall"],
                    input_weather["No of Rainy days"],
                    growth_week
                )

                results.append({
                    "SNo": i + 1,
                    "Date": forecast_date.strftime("%Y-%m-%d"),
                    "Crop Age (Days)": crop_age_days,
                    "Max Temp": input_weather["Max_Temp"],
                    "Min Temp": input_weather["Min_Temp"],
                    "Max Humidity": input_weather["Max_Humidity"],
                    "Min Humidity": input_weather["Min_Humidity"],
                    "Rainy Days": input_weather["No of Rainy days"],
                    "Rainfall": input_weather["Rainfall"],
                    "Predicted PDI (%)": f"{predicted_pdi:.2f}",
                    "Disease": f"{disease_name} ({severity_level})"
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

