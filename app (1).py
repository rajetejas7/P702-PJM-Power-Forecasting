import streamlit as st
import pandas as pd
import joblib

MODEL_FILE = "random_forest_pjm_model.joblib.gz"

FEATURES = [
    "Lag_1", "Lag_24", "Lag_168",
    "Hour", "DayOfWeek", "Month", "Holiday"
]

# Last 168 historical PJM demand values from the project dataset.
# These values provide the initial state for Lag_1, Lag_24 and Lag_168.
INITIAL_HISTORY = [
    5377.0000000000,
    5048.0000000000,
    4888.0000000000,
    4777.0000000000,
    4699.0000000000,
    4924.0000000000,
    5134.0000000000,
    5416.0000000000,
    5680.0000000000,
    5899.0000000000,
    6198.0000000000,
    6460.0000000000,
    6685.0000000000,
    6833.0000000000,
    7048.0000000000,
    7077.0000000000,
    7023.0000000000,
    6864.0000000000,
    6761.0000000000,
    6543.0000000000,
    6363.0000000000,
    6211.0000000000,
    5772.0000000000,
    5279.0000000000,
    4873.0000000000,
    4695.0000000000,
    4504.0000000000,
    4374.0000000000,
    4310.0000000000,
    4386.0000000000,
    4383.0000000000,
    4646.0000000000,
    5003.0000000000,
    5354.0000000000,
    5685.0000000000,
    5887.0000000000,
    6014.0000000000,
    6172.0000000000,
    6311.0000000000,
    6509.0000000000,
    6599.0000000000,
    6624.0000000000,
    6540.0000000000,
    6300.0000000000,
    6000.0000000000,
    5813.0000000000,
    5398.0000000000,
    4961.0000000000,
    4567.0000000000,
    4366.0000000000,
    4158.0000000000,
    4065.0000000000,
    4029.0000000000,
    4034.0000000000,
    4012.0000000000,
    4212.0000000000,
    4552.0000000000,
    4895.0000000000,
    5177.0000000000,
    5507.0000000000,
    5751.0000000000,
    5981.0000000000,
    6183.0000000000,
    6359.0000000000,
    6566.0000000000,
    6688.0000000000,
    6632.0000000000,
    6454.0000000000,
    6257.0000000000,
    6034.0000000000,
    5612.0000000000,
    5160.0000000000,
    4731.0000000000,
    4593.0000000000,
    4416.0000000000,
    4378.0000000000,
    4411.0000000000,
    4619.0000000000,
    4842.0000000000,
    5153.0000000000,
    5379.0000000000,
    5594.0000000000,
    5789.0000000000,
    5967.0000000000,
    6106.0000000000,
    6101.0000000000,
    6073.0000000000,
    6164.0000000000,
    6181.0000000000,
    6232.0000000000,
    6187.0000000000,
    6172.0000000000,
    6067.0000000000,
    5910.0000000000,
    5600.0000000000,
    5147.0000000000,
    4847.0000000000,
    4535.0000000000,
    4492.0000000000,
    4394.0000000000,
    4471.0000000000,
    4666.0000000000,
    4955.0000000000,
    5159.0000000000,
    5389.0000000000,
    5653.0000000000,
    5936.0000000000,
    6264.0000000000,
    6416.0000000000,
    6559.0000000000,
    6621.0000000000,
    6562.0000000000,
    6466.0000000000,
    6480.0000000000,
    6426.0000000000,
    6276.0000000000,
    6250.0000000000,
    6123.0000000000,
    5771.0000000000,
    5343.0000000000,
    5061.0000000000,
    4817.0000000000,
    4687.0000000000,
    4588.0000000000,
    4565.0000000000,
    4815.0000000000,
    5135.0000000000,
    5509.0000000000,
    5644.0000000000,
    5846.0000000000,
    6165.0000000000,
    6419.0000000000,
    6614.0000000000,
    6853.0000000000,
    6980.0000000000,
    7090.0000000000,
    7113.0000000000,
    7147.0000000000,
    7012.0000000000,
    6816.0000000000,
    6571.0000000000,
    6362.0000000000,
    5881.0000000000,
    5363.0000000000,
    5100.0000000000,
    4840.0000000000,
    4745.0000000000,
    4572.0000000000,
    4594.0000000000,
    4826.0000000000,
    5114.0000000000,
    5333.0000000000,
    5597.0000000000,
    5775.0000000000,
    6050.0000000000,
    6152.0000000000,
    6299.0000000000,
    6425.0000000000,
    6492.0000000000,
    6627.0000000000,
    6708.0000000000,
    6758.0000000000,
    6693.0000000000,
    6545.0000000000,
    6496.0000000000,
    6325.0000000000,
    5892.0000000000,
    5489.0000000000
]

LAST_HISTORICAL_DATETIME = pd.Timestamp("2018-08-03 00:00:00")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

def generate_forecast(model, hours=720):
    history = INITIAL_HISTORY.copy()

    future_dates = pd.date_range(
        start=LAST_HISTORICAL_DATETIME + pd.Timedelta(hours=1),
        periods=hours,
        freq="h"
    )

    predictions = []

    for dt in future_dates:
        X_future = pd.DataFrame([[
            history[-1],
            history[-24],
            history[-168],
            dt.hour,
            dt.dayofweek,
            dt.month,
            0
        ]], columns=FEATURES)

        prediction = model.predict(X_future)[0]
        predictions.append(prediction)
        history.append(prediction)

    forecast_df = pd.DataFrame({
        "Datetime": future_dates,
        "Forecast_MW": predictions
    })

    forecast_df["Date"] = forecast_df["Datetime"].dt.date

    daily_forecast = (
        forecast_df.groupby("Date")["Forecast_MW"]
        .agg(
            Average_MW="mean",
            Minimum_MW="min",
            Maximum_MW="max"
        )
        .reset_index()
    )

    return forecast_df, daily_forecast

st.set_page_config(
    page_title="PJM Energy Consumption Forecast",
    layout="wide"
)

st.title("PJM Hourly Energy Consumption Forecast")
st.subheader("P-702 | Final Random Forest Model")

st.write(
    "Forecast of PJM hourly energy consumption for the next 30 days "
    "using the final Random Forest model."
)

try:
    model = load_model()
    st.success("Random Forest model loaded successfully.")

    if st.button("Generate 30-Day Forecast"):
        with st.spinner("Generating 720 hourly forecasts..."):
            forecast_df, daily_forecast = generate_forecast(model, 720)

        st.success("720 hourly forecasts generated successfully.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Average Forecast", f"{forecast_df['Forecast_MW'].mean():,.2f} MW")
        c2.metric("Minimum Forecast", f"{forecast_df['Forecast_MW'].min():,.2f} MW")
        c3.metric("Maximum Forecast", f"{forecast_df['Forecast_MW'].max():,.2f} MW")

        st.subheader("Hourly Forecast")
        st.line_chart(
            forecast_df.set_index("Datetime")[["Forecast_MW"]]
        )

        st.subheader("Daily Forecast Summary")
        st.dataframe(daily_forecast, use_container_width=True)

        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Forecast CSV",
            data=csv,
            file_name="PJM_30_Day_Forecast.csv",
            mime="text/csv"
        )

except FileNotFoundError:
    st.error(
        "random_forest_pjm_model.joblib was not found. "
        "Keep it in the same GitHub repository as app.py."
    )
# ---------------------------------------------------------
# Streamlit interface
# ---------------------------------------------------------

st.set_page_config(
    page_title="PJM Energy Consumption Forecast",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ PJM Hourly Energy Consumption Forecast")
st.markdown("### P-702 | Final Random Forest Model")
st.write(
    "Forecast PJM hourly energy consumption for the next **30 days "
    "(720 hours)** using the final Random Forest model."
)

try:
    model = load_model()
    st.success("✅ Random Forest model loaded successfully.")

    if st.button("🚀 Generate 30-Day Forecast", type="primary"):

        with st.spinner("Generating 720 hourly forecasts..."):
            forecast_df, daily_forecast = generate_forecast(model, 720)

        st.success("✅ 720 hourly forecasts generated successfully.")

        # Forecast period
        forecast_start = forecast_df["Datetime"].min()
        forecast_end = forecast_df["Datetime"].max()

        st.markdown("### 📅 Forecast Period")
        p1, p2, p3 = st.columns(3)
        p1.metric("Forecast Start", forecast_start.strftime("%d-%b-%Y %H:%M"))
        p2.metric("Forecast End", forecast_end.strftime("%d-%b-%Y %H:%M"))
        p3.metric("Forecast Horizon", "720 Hours / 30 Days")

        # KPI cards
        st.markdown("### 📊 Forecast Highlights")

        avg_forecast = forecast_df["Forecast_MW"].mean()
        min_idx = forecast_df["Forecast_MW"].idxmin()
        max_idx = forecast_df["Forecast_MW"].idxmax()

        min_forecast = forecast_df.loc[min_idx, "Forecast_MW"]
        max_forecast = forecast_df.loc[max_idx, "Forecast_MW"]
        min_time = forecast_df.loc[min_idx, "Datetime"]
        max_time = forecast_df.loc[max_idx, "Datetime"]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Average Demand",
            f"{avg_forecast:,.2f} MW"
        )

        c2.metric(
            "Minimum Demand",
            f"{min_forecast:,.2f} MW",
            help=f"Predicted at {min_time.strftime('%d-%b-%Y %H:%M')}"
        )

        c3.metric(
            "Maximum Demand",
            f"{max_forecast:,.2f} MW",
            help=f"Predicted at {max_time.strftime('%d-%b-%Y %H:%M')}"
        )

        st.caption(
            f"Peak demand: **{max_forecast:,.2f} MW** on "
            f"**{max_time.strftime('%d-%b-%Y at %H:%M')}** | "
            f"Minimum demand: **{min_forecast:,.2f} MW** on "
            f"**{min_time.strftime('%d-%b-%Y at %H:%M')}**"
        )

        # Tabs keep the UI clean
        tab1, tab2, tab3 = st.tabs(
            ["📈 Hourly Forecast", "📅 Daily Summary", "⬇️ Download"]
        )

        with tab1:
            st.subheader("Hourly Energy Consumption Forecast")
            st.line_chart(
                forecast_df.set_index("Datetime")[["Forecast_MW"]],
                height=450
            )

        with tab2:
            st.subheader("Daily Forecast Summary")
            st.dataframe(
                daily_forecast,
                use_container_width=True,
                hide_index=True
            )

        with tab3:
            st.subheader("Download Forecast Results")

            csv = forecast_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download 720-Hour Forecast CSV",
                data=csv,
                file_name="PJM_30_Day_Forecast.csv",
                mime="text/csv"
            )

            st.info(
                "The downloaded CSV contains the complete hourly forecast "
                "for the 30-day forecasting horizon."
            )

        st.divider()
        st.caption(
            "P-702 | PJM Hourly Energy Consumption Forecast | "
            "Final Random Forest Model"
        )

except FileNotFoundError:
    st.error(
        "random_forest_pjm_model.joblib.gz was not found. "
        "Keep it in the same GitHub repository as app.py."
    )
