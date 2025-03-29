import pandas as pd
import numpy as np
from scipy.optimize import minimize


def power_model(params, data, wind_speed_threshold):
    """
    Model to predict power output based on parameters.

    Args:
        params: List of [yaw_angle, pitch_angle]
        data: DataFrame with wind turbine data
        wind_speed_threshold: Wind speed threshold for optimization

    Returns:
        Negative power output (for minimization)
    """
    yaw_angle, pitch_angle = params

    # Filter data for relevant wind speeds
    relevant_data = data[data['wind_speed'] >= wind_speed_threshold].copy()

    if len(relevant_data) == 0:
        return 0

    # Calculate yaw misalignment
    relevant_data['yaw_misalignment'] = np.abs(relevant_data['wind_direction'] - yaw_angle)
    # Adjust for circular nature of angles
    relevant_data.loc[relevant_data['yaw_misalignment'] > 180, 'yaw_misalignment'] = 360 - relevant_data.loc[
        relevant_data['yaw_misalignment'] > 180, 'yaw_misalignment']

    # Calculate pitch adjustment
    relevant_data['pitch_adjustment'] = np.abs(pitch_angle - relevant_data['pitch_angle'])

    # Simplified power model based on wind speed, yaw misalignment, and pitch
    # This is a simplified model and should be replaced with a more accurate one based on turbine specifications
    power = relevant_data['wind_speed'] ** 3 * 0.5 * relevant_data['air_density'] * 0.5 * \
            (1 - 0.01 * relevant_data['yaw_misalignment']) * \
            (1 - 0.02 * relevant_data['pitch_adjustment'])

    return -np.mean(power)  # Negative for minimization


def optimize_power(data, wind_speed_threshold=12.0, yaw_angle_range=15, pitch_angle_range=10):
    """
    Optimize turbine parameters for maximum power output.

    Args:
        data: DataFrame with wind turbine data
        wind_speed_threshold: Wind speed threshold for optimization
        yaw_angle_range: Range for yaw angle optimization (± degrees)
        pitch_angle_range: Range for pitch angle optimization (± degrees)

    Returns:
        Dictionary of optimized parameters and expected gain
    """
    # Calculate average wind direction
    avg_wind_direction = data['wind_direction'].mean()

    # Calculate average pitch angle
    avg_pitch_angle = data['pitch_angle'].mean()

    # Initial parameters
    initial_params = [avg_wind_direction, avg_pitch_angle]

    # Parameter bounds
    bounds = [
        (avg_wind_direction - yaw_angle_range, avg_wind_direction + yaw_angle_range),
        (max(0, avg_pitch_angle - pitch_angle_range), avg_pitch_angle + pitch_angle_range)
    ]

    # Optimization
    result = minimize(
        power_model,
        initial_params,
        args=(data, wind_speed_threshold),
        bounds=bounds,
        method='L-BFGS-B'
    )

    # Calculate expected power gain
    initial_power = -power_model(initial_params, data, wind_speed_threshold)
    optimized_power = -result.fun
    power_gain_percent = (optimized_power / initial_power - 1) * 100 if initial_power > 0 else 0

    return {
        'yaw_angle': result.x[0],
        'pitch_angle': result.x[1]
    }, power_gain_percent
