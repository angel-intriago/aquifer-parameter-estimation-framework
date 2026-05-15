import os
import warnings
from datetime import datetime
from src.data_loading import load_hydrogeological_data
from src.advanced_hydro_calibrator import AdvancedHydrogeologicalCalibrator

warnings.filterwarnings("ignore", category=RuntimeWarning, module="seaborn")

def main():
    """
    Main workflow for advanced hydrogeological calibration.
    """
    print("====== STARTING ADVANCED HYDROGEOLOGICAL CALIBRATION SCRIPT ======")

    print(f"Working directory: {os.getcwd()}")

    # Point to demo_data/ by default; replace with 'data' for the full dataset
    DATA_DIR = 'demo_data'
    data_paths = {
        'observed_df_path': os.path.join(DATA_DIR, 'head_time_series.csv'),
        'private_wells_path': os.path.join(DATA_DIR, 'private_supply_wells.shp'),
        'public_wells_path': os.path.join(DATA_DIR, 'public_supply_wells.shp'),
        'private_stratigraphy_path': os.path.join(DATA_DIR, 'stratigraphy_private_wells.csv'),
        'public_stratigraphy_path': os.path.join(DATA_DIR, 'stratigraphy_public_wells.csv')
    }

    try:
        loaded_data = load_hydrogeological_data(**data_paths)
    except Exception as e:
        print(f"CRITICAL ERROR LOADING DATA: {e}")
        return

    # Generate automatic run name based on date and time
    current_run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define parameters for the run
    run_params = {
        'run_name': current_run_name,
        'operation_mode': 'complete', # Options: 'complete', 'filtering_only'
        'consistency_iterations': 1,
        'calibration_method': 'differential_evolution',
        'consider_interference': False,
        'generate_event_plots': False,
        'generate_summary_plots': False,
        'filter_slope_outliers': True,
        'min_r_squared_selection': 0.95,
        'min_drawdown_ratio': 0.5,
        'min_events_per_well': 3,
        'max_delta_t_minutes': 15,
        'calibrate_target_well_flow': True,
        'calibrate_public_well_flows': True,
        'calibrate_private_well_flows': True,
        'target_uncertainty': (0.8, 1.2),
        'public_uncertainty': (0.9, 1.1),
        'private_uncertainty': (0.33, 3.0),

        # Hydraulic conductivity (K) ranges for each geological formation (m/d)
        'conductivity_ranges': {
            'Fm Serra Geral': (1e-5, 1e-3),
            'Fm Botucatu': (0.1, 15.0),
            'Fm Guará Sup': (0.5, 5.0),
            'Fm Guará Inf': (0.2, 3.0),
            'Fm Piramboia Sup': (0.01, 4.0),
            'Fm Piramboia Inf': (0.01, 2.0),
            'Fm Diabásio': (1e-5, 1e-3),
            'Fm Corumbataí': (0.01, 0.5)
        },

        # Specific uncertainty ranges for neighborhoods
        'neighborhood_uncertainty': {}
    }

    # Initialize the calibrator
    calibrator = AdvancedHydrogeologicalCalibrator(**run_params)
    
    # Set data to the calibrator
    calibrator.set_data(loaded_data)

    # Execute workflow according to operation mode
    calibrator.execute()

    print("\n====== ADVANCED CALIBRATION SCRIPT FINISHED ======")

if __name__ == '__main__':
    main()
