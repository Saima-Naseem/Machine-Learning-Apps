# -----------------------------------------
# Import Required Libraries
# -----------------------------------------

import streamlit as st
import pandas as pd
import joblib
# -----------------------------------------
# Configure Streamlit Page
# -----------------------------------------

st.set_page_config(
    page_title="Pakistan Budget Estimates Forecaster",
    layout="wide"
)
st.markdown("""
<style>

/* ===== Glass Sidebar ===== */

[data-testid="stSidebar"]{
    background: rgba(22, 34, 57, 0.55);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border-right: 1px solid rgba(255,255,255,0.12);
}

[data-testid="stSidebar"] > div:first-child{
    background: transparent;
}

[data-testid="stSidebar"] *{
    color:white;
}

</style>
""", unsafe_allow_html=True)
# -----------------------------------------
# App Title
# -----------------------------------------

st.title("Pakistan Budget Estimates Forecaster")

# -----------------------------------------
# Load Trained Model
# -----------------------------------------

import os

st.write("Current Working Directory:", os.getcwd())
st.write("Files:", os.listdir())

model = joblib.load("pakistan_budget_predictor.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.success("✅ Model loaded successfully")
st.success("✅ Feature columns loaded successfully")
# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Pakistan Budget Predictor")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 About")

st.sidebar.write(
    """
Forecast Pakistan's future budget estimates using a Machine Learning
Linear Regression model trained on historical budget data.
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Model")

st.sidebar.write("**Algorithm:** Linear Regression")
st.sidebar.write("**Input Features:**")
st.sidebar.write("• Year")
st.sidebar.write("• Previous Year's Total Budget")

st.sidebar.write("**Target:**")
st.sidebar.write("• Total Budget Estimate")

st.sidebar.markdown("---")

st.sidebar.subheader("📅 Forecast")

st.sidebar.write("Available Years")
st.sidebar.success("2027 – 2036")

st.sidebar.markdown("---")

st.sidebar.subheader("⚠️ Disclaimer")

st.sidebar.caption(
    """
Forecasts are generated using historical budget trends. The model was trained using Pakistan's federal budget total figures spanning 1973 to 2026.
It is based on the assumption that historical trends will continue into the future.
At present, budget forecasts are intended for educational and analytical purposes.
"""
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
<div style="text-align:center; padding-top:10px;">
    <span style="font-size:14px; color:#D3D3D3;">Developed by</span><br>
    <span style="font-size:18px; font-weight:600;">Saima Naseem</span><br>
    <span style="font-size:13px; color:#B8C6DB;">
        Connecting AI, Governance & Public Finance
    </span>
</div>
""",
unsafe_allow_html=True
)
# -----------------------------------------
# Budget Structures
# -----------------------------------------

budget_structures = {

    "2026 Budget Structure": {
        "Defense_Percentage": 13.83,
        "Education_Percentage": 9.76,
        "Health_Percentage": 4.68,
        "Infrastructure_Percentage": 20.02,
        "Agriculture_Percentage": 10.01,
        "State_Transfers_Percentage": 10.01,
        "Social_Welfare_Percentage": 10.01,
        "Administration_and_Others_Percentage": 16.68
    }

}

with st.expander("📖 How to Use"):
    st.write("""
1. Enter the previous year's total budget (Billion USD).
2. Select the target prediction year.
3. Click **Generate Budget Forecast**.
4. The app automatically forecasts all intermediate years up to the selected year.
5. View the predicted budget, expected growth, and forecast table.
""")
st.caption(
    "Example: To forecast the 2030 budget, enter the 2026 total budget and select 2030. The app will automatically predict the budgets for 2027, 2028, 2029, and then 2030."
)

# -----------------------------------------
# User Inputs
# -----------------------------------------

st.subheader("User Inputs")

selected_year = st.selectbox(
    "Forecast Year",
    list(range(2027, 2037))
)

previous_budget = st.number_input(
    "Previous Year's Total Budget (Billion USD)",
    min_value=0.0,
    value=60.90,
    step=0.1
)

# -----------------------------------------
# Prediction
# -----------------------------------------

if st.button("Generate Budget Forecast"):

    forecast = []
    current_budget = previous_budget

    # Forecast year by year
    for year in range(2027, selected_year + 1):

        input_data = pd.DataFrame([{
            "Year": year,
            "Previous_Total_Budget": current_budget
        }])

        input_data = input_data[feature_columns]

        predicted_budget = model.predict(input_data)[0]

        growth = (
            (predicted_budget - current_budget)
            / current_budget
        ) * 100

        forecast.append({
            "Year": year,
            "Predicted Budget (Billion USD)": round(predicted_budget, 2),
            "Expected Growth (%)": round(growth, 2)
        })

        # Use this prediction as the previous year's budget
        current_budget = predicted_budget

    # Create forecast table
    forecast_df = pd.DataFrame(forecast)

    # Final prediction (selected year)
    final_prediction = forecast_df.iloc[-1]

    st.success("Budget forecast generated successfully.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Budget (Billion USD)",
            f"{final_prediction['Predicted Budget (Billion USD)']:.2f}"
        )

    with col2:
        st.metric(
            "Expected Growth (%)",
            f"{final_prediction['Expected Growth (%)']:.2f}%"
        )

    st.subheader(f"Budget Forecast: 2027–{selected_year}")

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The forecast is generated sequentially. Each year's predicted budget is used as the previous year's budget to forecast the next year."
    
    )
    
