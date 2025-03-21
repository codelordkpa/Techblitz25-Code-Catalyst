# import streamlit as st
# import joblib
# import pandas as pd
# import plotly.express as px

# # Load trained model and preprocessing objects
# model = joblib.load('air_quality_rf_model.pkl')
# scaler = joblib.load('scaler.pkl')
# le = joblib.load('label_encoder.pkl')

# # Streamlit webpage
# def main():
#     st.title("🌿 Air Quality Prediction App")

#     st.sidebar.header('Input Environmental Parameters')

#     # User inputs
#     Temperature = st.sidebar.slider('Temperature (°C)', 0, 50, 25)
#     Humidity = st.sidebar.slider('Humidity (%)', 0, 100, 50)
#     PM25 = st.sidebar.slider('PM2.5 (µg/m³)', 0, 300, 30)
#     PM10 = st.sidebar.slider('PM10 (µg/m³)', 0, 400, 50)
#     NO2 = st.sidebar.slider('NO2 (ppb)', 0, 100, 20)
#     SO2 = st.sidebar.slider('SO2 (ppb)', 0, 100, 10)
#     CO = st.sidebar.slider('CO (ppm)', 0.0, 10.0, 1.0)
#     Proximity_to_Industrial_Areas = st.sidebar.slider('Proximity to Industrial Areas (km)', 0.1, 20.0, 5.0)
#     Population_Density = st.sidebar.slider('Population Density (people/km²)', 100, 5000, 1000)

#     input_data = pd.DataFrame({
#         'Temperature': [Temperature],
#         'Humidity': [Humidity],
#         'PM2.5': [PM25],
#         'PM10': [PM10],
#         'NO2': [NO2],
#         'SO2': [SO2],
#         'CO': [CO],
#         'Proximity_to_Industrial_Areas': [Proximity_to_Industrial_Areas],
#         'Population_Density': [Population_Density]
#     })

#     # Prediction
#     input_scaled = scaler.transform(input_data)
#     prediction = model.predict(input_scaled)
#     prediction_label = le.inverse_transform(prediction)[0]

#     st.subheader('✅ Prediction Result:')
#     st.markdown(f'### Air Quality Level: **{prediction_label}**')

#     # Interactive graph
#     fig = px.bar(
#         input_data.melt(),
#         x='variable', y='value',
#         color='variable',
#         labels={'variable': 'Feature', 'value': 'Value'},
#         title="Input Feature Values"
#     )

#     st.plotly_chart(fig, use_container_width=True)

# if __name__ == '__main__':
#     main()


import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import base64

# Load trained model and preprocessing objects
model = joblib.load('air_quality_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
df = pd.read_csv('TechBlitz DataScience Dataset.csv')

# Streamlit webpage
def main():
    st.title("🌿 Air Quality Prediction App")

    st.sidebar.header('Input Environmental Parameters')

    # User inputs
    Temperature = st.sidebar.slider('Temperature (°C)', 0, 50, 25)
    Humidity = st.sidebar.slider('Humidity (%)', 0, 100, 50)
    PM25 = st.sidebar.slider('PM2.5 (µg/m³)', 0, 300, 30)
    PM10 = st.sidebar.slider('PM10 (µg/m³)', 0, 400, 50)
    NO2 = st.sidebar.slider('NO2 (ppb)', 0, 100, 20)
    SO2 = st.sidebar.slider('SO2 (ppb)', 0, 100, 10)
    CO = st.sidebar.slider('CO (ppm)', 0.0, 10.0, 1.0)
    Proximity_to_Industrial_Areas = st.sidebar.slider('Proximity to Industrial Areas (km)', 0.1, 20.0, 5.0)
    Population_Density = st.sidebar.slider('Population Density (people/km²)', 100, 5000, 1000)

    input_data = pd.DataFrame({
        'Temperature': [Temperature],
        'Humidity': [Humidity],
        'PM2.5': [PM25],
        'PM10': [PM10],
        'NO2': [NO2],
        'SO2': [SO2],
        'CO': [CO],
        'Proximity_to_Industrial_Areas': [Proximity_to_Industrial_Areas],
        'Population_Density': [Population_Density]
    })

    # Prediction
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction_label = le.inverse_transform(prediction)[0]

    st.subheader('✅ Prediction Result:')
    st.markdown(f'### Air Quality Level: **{prediction_label}**')

    # Interactive graph for input
    fig = px.bar(
        input_data.melt(),
        x='variable', y='value',
        color='variable',
        labels={'variable': 'Feature', 'value': 'Value'},
        title="Input Feature Values"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("📊 Data Visualizations")

    # Bar Chart
    st.subheader("Air Quality Distribution")
    st.bar_chart(df['Air Quality'].value_counts())

    # Pie Chart
    st.subheader("Air Quality Proportions")
    pie_data = df['Air Quality'].value_counts().reset_index()
    pie_data.columns = ['Air Quality', 'Count']
    fig_pie = px.pie(pie_data, names='Air Quality', values='Count', title='Air Quality Distribution')
    st.plotly_chart(fig_pie)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df.drop('Air Quality', axis=1).corr()
    fig_heat = px.imshow(corr, text_auto=True, aspect='auto', title="Feature Correlation")
    st.plotly_chart(fig_heat)

    # Boxplots
    st.subheader("Boxplots of Key Features by Air Quality")
    for col in ['PM2.5', 'PM10', 'Temperature']:
        fig_box = px.box(df, x='Air Quality', y=col, title=f'{col} vs Air Quality')
        st.plotly_chart(fig_box)

    # Scatter Matrix
    st.subheader("Scatter Matrix")
    fig_scat = px.scatter_matrix(df, dimensions=['PM2.5', 'PM10', 'NO2', 'SO2', 'CO'], color='Air Quality')
    st.plotly_chart(fig_scat)

    # Radar Chart
    st.subheader("Radar Plot")
    radar_df = df.groupby('Air Quality').mean().reset_index()
    categories = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
    fig_radar = go.Figure()
    for _, row in radar_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name=row['Air Quality']
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Radar Plot of Mean Features")
    st.plotly_chart(fig_radar)

    # Parallel Coordinates
    st.subheader("Parallel Coordinates")
    fig_parallel = px.parallel_coordinates(
        df, dimensions=categories, color=df['Air Quality'].astype('category').cat.codes,
        color_continuous_scale=px.colors.diverging.Tealrose
    )
    st.plotly_chart(fig_parallel)

    # 3D Scatter Plot
    st.subheader("3D Scatter Plot")
    fig_3d = px.scatter_3d(
        df, x='PM2.5', y='PM10', z='NO2', color='Air Quality', size='Temperature',
        title='3D Scatter Plot of Air Quality Factors'
    )
    st.plotly_chart(fig_3d)

    # Animated Bubble Chart
    st.subheader("Animated Bubble Chart")
    fig_bubble = px.scatter(
        df, x='Proximity_to_Industrial_Areas', y='Population_Density',
        size='PM2.5', color='Air Quality', animation_frame='Temperature',
        size_max=60, range_x=[0, 20], range_y=[0, 6000]
    )
    st.plotly_chart(fig_bubble)

    # Animated Feature Importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': df.drop('Air Quality', axis=1).columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance')
    fig_feat = px.bar(
        importance_df, x='Importance', y='Feature', orientation='h',
        title='Feature Importance (Random Forest)', color='Importance'
    )
    st.plotly_chart(fig_feat)

if __name__ == '__main__':
    main()
