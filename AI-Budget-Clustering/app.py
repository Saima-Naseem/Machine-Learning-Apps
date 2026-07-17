import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ----------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------

st.set_page_config(
    page_title="AI Budget Cluster",
    page_icon="📊",
    layout="centered"
)
st.markdown("""
<style>

/* Sidebar width */
section[data-testid="stSidebar"]{
    width:220px !important;
}

section[data-testid="stSidebar"] > div{
    width:220px !important;
}

/* Reduce top and bottom padding */
section[data-testid="stSidebar"] .block-container{
    padding-top:0.6rem !important;
    padding-bottom:0.5rem !important;
    padding-left:0.8rem !important;
    padding-right:0.8rem !important;
}

/* Reduce spacing between widgets */
div[data-testid="stVerticalBlock"] > div{
    gap:0.3rem !important;
}

/* Reduce space around markdown */
div[data-testid="stMarkdownContainer"] p{
    margin-bottom:0.2rem !important;
}

/* Make radio buttons more compact */
div[role="radiogroup"] > label{
    padding-top:0.15rem !important;
    padding-bottom:0.15rem !important;
}

/* Make horizontal separators tighter */
hr{
    margin-top:0.4rem !important;
    margin-bottom:0.4rem !important;
}

</style>
""", unsafe_allow_html=True)
# ----------------------------------------
# LOAD FILES
# ----------------------------------------

@st.cache_resource
def load_models():

    model = joblib.load("budget_cluster_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    cluster_names = joblib.load("cluster_names.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    return model, scaler, pca, cluster_names, feature_columns


@st.cache_data
def load_data():

    df = pd.read_csv("country_budget_profiles.csv")

    return df


model, scaler, pca, cluster_names, feature_columns = load_models()

data = load_data()
# ----------------------------------------
# SIDEBAR
# ----------------------------------------
st.sidebar.title("📊 AI Budget Cluster")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "🧭 Navigation",
    [
        "🏠 Home",
        "🌍 Country Analysis",
        "📈 PCA Visualization"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Evaluation")

st.sidebar.metric("Silhouette Score", "0.259")
st.sidebar.metric("Calinski-Harabasz", "12.65")
st.sidebar.metric("Davies-Bouldin", "1.23")

st.sidebar.write(
    """
This application groups countries according to
their national budget allocation patterns using
a trained K-Means Machine Learning model.
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")

st.sidebar.write("Algorithm : K-Means Clustering")
st.sidebar.write(f"Countries : {len(data)}")
st.sidebar.write(f"Features : {len(feature_columns)}")
st.sidebar.write(f"Clusters : {len(cluster_names)}")

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

# ----------------------------------------
# MAIN PAGE
# ----------------------------------------

if page == "🏠 Home":

    st.title("📊 AI Budget Cluster Dashboard")

    st.write("""
Welcome to the AI Budget Cluster Dashboard.

This application uses Machine Learning (K-Means Clustering) to group countries according to their budget allocation patterns.

Use the navigation panel on the left to explore different sections of the dashboard.
""")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌍 Countries", len(data))
    col2.metric("📊 Clusters", len(cluster_names))
    col3.metric("📈 Features", len(feature_columns))
    col4.metric("🤖 Algorithm", "K-Means")

    st.markdown("---")
    
# ----------------------------------------
# COUNTRY SELECTION
# ----------------------------------------

country_column = "Country"

countries = sorted(data[country_column].unique())

selected_country = st.selectbox(
    
    "🌍 Select a Country",
    countries
)

country_data = data[data[country_column] == selected_country]

st.markdown("---")
# ----------------------------------------
# PREDICT BUTTON
# ----------------------------------------

if st.button("🔍 Predict Budget Cluster"):

    # Get feature values
    X = country_data[feature_columns]

    # Scale data
    X_scaled = scaler.transform(X)

    # Predict cluster
    predicted_cluster = int(model.predict(X_scaled)[0])

    # Get cluster name
    try:
        cluster_name = cluster_names[predicted_cluster]
    except:
        cluster_name = f"Cluster {predicted_cluster}"

    # ----------------------------------------
    # RESULT
    # ----------------------------------------

    st.success("Prediction Completed")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Cluster",
            predicted_cluster
        )

    with col2:
        st.metric(
            "Cluster Profile",
            cluster_name
        )

    cluster_descriptions = {
        0: "Balanced Allocation",
        1: "Human Development",
        2: "Defense Priority",
        3: "Infrastructure Growth",
        4: "Social Investment",
        5: "Agricultural Focus",
        6: "Diversified Economy"
    }

    st.caption(
        f"Profile: {cluster_descriptions.get(predicted_cluster, 'Budget Pattern')}"
    )

    st.markdown("---")

    # ----------------------------------------
    # BUDGET PROFILE
    # ----------------------------------------

    st.subheader("📋 Budget Allocation Profile (%)")
    budget_profile = X.T.copy()
    budget_profile.columns = ["Percentage"]

    st.dataframe(
        budget_profile.style.format("{:.2f}")
    )

        # ----------------------------------------
    # INTERACTIVE BUDGET BAR CHART
    # ----------------------------------------

    st.subheader("📊 Budget Allocation Chart")

    budget_df = pd.DataFrame({
        "Category": feature_columns,
        "Percentage": X.iloc[0].values
    })

    fig = px.bar(
        budget_df,
        x="Category",
        y="Percentage",
        text="Percentage",
        title=f"{selected_country} Budget Allocation"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside"
    )

    fig.update_layout(
        height=350,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Budget Categories",
        yaxis_title="Budget Percentage",
        xaxis_tickangle=-45,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=40)
    )

    st.plotly_chart(
        fig,
        use_container_width=False,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "pan2d",
                "select2d",
                "lasso2d",
                "autoScale2d",
                "resetScale2d",
                "hoverCompareCartesian",
                "toggleSpikelines"
            ]
        }
    )

    st.markdown("---")
    # ----------------------------------------
# INTERACTIVE PCA VISUALIZATION
# ----------------------------------------

st.subheader("🌍 Interactive PCA Cluster Visualization")

# Scale all countries
all_scaled = scaler.transform(data[feature_columns])

# PCA transformation
all_pca = pca.transform(all_scaled)

# Predict clusters
all_clusters = model.predict(all_scaled)

# Create dataframe for Plotly
pca_df = pd.DataFrame({
    "PC1": all_pca[:, 0],
    "PC2": all_pca[:, 1],
    "Country": data["Country"],
    "Cluster": all_clusters.astype(str)
})
# Create interactive scatter plot
fig = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_name="Country",
    title="Interactive PCA Cluster Visualization"
)
fig.for_each_trace(
    lambda t: t.update(
        name={
            "0": "0 - Balanced Allocation",
            "1": "1 - Human Development",
            "2": "2 - Defense Priority",
            "3": "3 - Infrastructure Growth",
            "4": "4 - Social Investment",
            "5": "5 - Agricultural Focus",
            "6": "6 - Diversified Economy"
        }.get(t.name, t.name),
        legendgroup={
            "0": "0 - Balanced Allocation",
            "1": "1 - Human Development",
            "2": "2 - Defense Priority",
            "3": "3 - Infrastructure Growth",
            "4": "4 - Social Investment",
            "5": "5 - Agricultural Focus",
            "6": "6 - Diversified Economy"
        }.get(t.name, t.name)
    )
)

# Highlight selected country
selected_point = pca_df[pca_df["Country"] == selected_country]

fig.add_trace(
    go.Scatter(
        x=selected_point["PC1"],
        y=selected_point["PC2"],
        mode="markers+text",
        text=selected_point["Country"],
        textposition="top center",
        marker=dict(
            symbol="star",
            size=18,
            color="red",
            line=dict(color="black", width=1)
        ),
        name="Selected Country"
    )
)

fig.update_layout(
    height=500,
    legend_title="Cluster"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "AI_Budget_Cluster_PCA",
            "height": 600,
            "width": 900,
            "scale": 2
        },
        "modeBarButtonsToRemove": [
            "pan2d",
            "select2d",
            "lasso2d",
            "autoScale2d",
            "resetScale2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "toggleSpikelines"
        ]
    }
)