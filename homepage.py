import streamlit as st


st.set_page_config(
    page_title="Use Pygwalker In Streamlit",
    layout="wide"
)

visualized_page = st.Page("visualize_data.py", title="Visualizations", icon="📊", default=True)
model_page = st.Page("model_prediction.py", title="Model Prediction", icon="🧠")


pg = st.navigation(
    {
        "Visualizations": [visualized_page],
        "Model Prediction": [model_page]
    }
)

pg.run()