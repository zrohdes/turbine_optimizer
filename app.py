import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from power_optimizer import optimize_power
from data_processor import process_data, calculate_metrics

# Page configuration
st.set_page_config(
    page_title="Wind Turbine Power Optimizer",
    page_icon="🌬️",
    layout="wide"
)

# App title and description
st.title("Wind Turbine Power Optimization")
st.markdown("""
This application analyzes wind turbine data and provides recommendations 
for optimizing power generation based on environmental conditions.
""")

# Sidebar for data upload and parameters
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload wind turbine data (CSV)", type=["csv"])

# Parameter settings
st.sidebar.header("Optimization Parameters")
wind_speed_threshold = st.sidebar.slider("Wind Speed Threshold (m/s)", 3.0, 25.0, 12.0)
yaw_angle_range = st.sidebar.slider("Yaw Angle Range (±degrees)", 0, 30, 15)
pitch_angle_range = st.sidebar.slider("Pitch Angle Range (±degrees)", 0, 20, 10)

# Main content area
if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(data.head())

    # Process data
    processed_data = process_data(data)

    # Data visualization
    st.subheader("Data Visualization")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Wind Speed vs Power Output")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='wind_speed', y='power_output', data=processed_data, alpha=0.6)
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Power Output (kW)')
        st.pyplot(fig)

    with col2:
        st.write("Power Output Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(processed_data['power_output'], kde=True)
        plt.xlabel('Power Output (kW)')
        plt.ylabel('Frequency')
        st.pyplot(fig)

    # Optimization
    st.subheader("Power Optimization")

    if st.button("Run Optimization"):
        with st.spinner("Optimizing power generation..."):
            optimized_params, expected_gain = optimize_power(
                processed_data,
                wind_speed_threshold=wind_speed_threshold,
                yaw_angle_range=yaw_angle_range,
                pitch_angle_range=pitch_angle_range
            )

            # Display optimization results
            st.success(f"Optimization complete! Expected power gain: {expected_gain:.2f}%")

            col1, col2, col3 = st.columns(3)
            col1.metric("Optimal Yaw Angle", f"{optimized_params['yaw_angle']:.2f}°")
            col2.metric("Optimal Pitch Angle", f"{optimized_params['pitch_angle']:.2f}°")
            col3.metric("Estimated Power Increase", f"{expected_gain:.2f}%")

            # Visualize before/after
            st.subheader("Before vs After Optimization")

            # Calculate metrics
            current_metrics, optimized_metrics = calculate_metrics(processed_data, optimized_params)

            comparison_data = pd.DataFrame({
                'Metric': ['Average Power (kW)', 'Efficiency (%)', 'Annual Energy Production (MWh)'],
                'Current': [current_metrics['avg_power'], current_metrics['efficiency'],
                            current_metrics['annual_energy']],
                'Optimized': [optimized_metrics['avg_power'], optimized_metrics['efficiency'],
                              optimized_metrics['annual_energy']],
                'Improvement (%)': [
                    (optimized_metrics['avg_power'] / current_metrics['avg_power'] - 1) * 100,
                    (optimized_metrics['efficiency'] / current_metrics['efficiency'] - 1) * 100,
                    (optimized_metrics['annual_energy'] / current_metrics['annual_energy'] - 1) * 100
                ]
            })

            st.table(comparison_data.set_index('Metric'))
else:
    st.info("Upload a CSV file to begin analysis and optimization")

    # Sample data description
    st.subheader("Expected Data Format")
    st.markdown("""
    Your CSV should include the following columns:
    - `timestamp`: Date and time of measurement
    - `wind_speed`: Wind speed in m/s
    - `wind_direction`: Wind direction in degrees
    - `temperature`: Ambient temperature in °C
    - `air_density`: Air density in kg/m³
    - `power_output`: Power output in kW
    - `yaw_angle`: Current yaw angle in degrees
    - `pitch_angle`: Current pitch angle in degrees

    Example:
    """)

    # Create example dataframe
    example_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='H'),
        'wind_speed': [7.2, 8.5, 6.3, 9.1, 5.2],
        'wind_direction': [245, 250, 255, 240, 260],
        'temperature': [12.3, 11.8, 12.0, 11.5, 12.5],
        'air_density': [1.225, 1.220, 1.223, 1.218, 1.226],
        'power_output': [1250, 1650, 980, 1820, 750],
        'yaw_angle': [247, 253, 256, 242, 258],
        'pitch_angle': [2.5, 3.0, 2.2, 3.5, 2.0]
    })

    st.dataframe(example_data)

# Show app information in footer
st.markdown("---")
st.markdown("©️ 2025 Wind Turbine Power Optimizer")
