import pandas as pd
import numpy as np


def process_data(data):
    """
    Process and clean wind turbine data.

    Args:
        data: Raw DataFrame with wind turbine data

    Returns:
        Processed DataFrame
    """
    processed_data = data.copy()

    # Convert timestamp to datetime if it's not already
    if 'timestamp' in processed_data.columns and not pd.api.types.is_datetime64_any_dtype(processed_data['timestamp']):
        processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'])

    # Handle missing values
    processed_data = processed_data.interpolate(method='linear')

    # Filter out unrealistic values
    processed_data = processed_data[processed_data['wind_speed'] >= 0]
    processed_data = processed_data[processed_data['wind_speed'] <= 50]  # Max realistic wind speed

    if 'power_output' in processed_data.columns:
        processed_data = processed_data[processed_data['power_output'] >= 0]

    # Add derived features
    if 'wind_speed' in processed_data.columns and 'air_density' in processed_data.columns:
        # Theoretical power in the wind (P = 0.5 * ρ * A * v³)
        # Assuming a standard turbine area
        turbine_area = 10000  # m², can be adjusted based on actual turbine size
        processed_data['theoretical_power'] = 0.5 * processed_data['air_density'] * turbine_area * processed_data[
            'wind_speed'] ** 3

        # Calculate efficiency if power_output is available
        if 'power_output' in processed_data.columns:
            processed_data['efficiency'] = (processed_data['power_output'] * 1000) / processed_data[
                'theoretical_power']  # Convert kW to W

            # Cap efficiency at realistic values
            processed_data['efficiency'] = processed_data['efficiency'].clip(0, 0.59)  # Betz limit is 0.59

    return processed_data


def calculate_metrics(data, optimized_params):
    """
    Calculate performance metrics for current and optimized settings.

    Args:
        data: Processed DataFrame with wind turbine data
        optimized_params: Dictionary with optimized parameters

    Returns:
        Two dictionaries with current and optimized metrics
    """
    # Current metrics
    current_metrics = {
        'avg_power': data['power_output'].mean(),
        'efficiency': data['efficiency'].mean() * 100 if 'efficiency' in data.columns else 35,
        'annual_energy': data['power_output'].mean() * 8760 / 1000  # MWh (assuming 8760 hours in a year)
    }

    # Create a copy for optimized metrics
    optimized_data = data.copy()

    # Apply simple improvement model based on optimized parameters
    # This is a simplified model and should be replaced with a more accurate one
    yaw_improvement = 1 + 0.003 * np.mean(np.abs(data['wind_direction'] - optimized_params['yaw_angle']))
    pitch_improvement = 1 + 0.002 * np.mean(np.abs(data['pitch_angle'] - optimized_params['pitch_angle']))
    total_improvement = yaw_improvement * pitch_improvement

    # Calculate optimized metrics
    optimized_metrics = {
        'avg_power': current_metrics['avg_power'] * total_improvement,
        'efficiency': min(current_metrics['efficiency'] * total_improvement, 59),  # Cap at Betz limit
        'annual_energy': current_metrics['annual_energy'] * total_improvement
    }

    return current_metrics, optimized_metrics
