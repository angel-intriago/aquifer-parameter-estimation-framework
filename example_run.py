import os
from src.data_loading import load_hydrogeological_data
from src.advanced_hydro_calibrator import AdvancedHydrogeologicalCalibrator

def run_example():
    """
    Example script to run a quick test of the hydrogeological calibration framework.
    This script runs in 'filtering_only' mode to quickly verify the pipeline.
    """
    print("====== RUNNING QUICK TEST EXAMPLE ======")

    # Ensure data paths are correct (relative to the repository root)
    data_paths = {
        'observed_df_path': os.path.join('demo_data', 'head_time_series.csv'),
        'private_wells_path': os.path.join('demo_data', 'private_supply_wells.shp'),
        'public_wells_path': os.path.join('demo_data', 'public_supply_wells.shp'),
        'private_stratigraphy_path': os.path.join('demo_data', 'stratigraphy_private_wells.csv'),
        'public_stratigraphy_path': os.path.join('demo_data', 'stratigraphy_public_wells.csv')
    }

    # Verify if data exists
    for key, path in data_paths.items():
        if not os.path.exists(path):
            print(f"ERROR: Data file not found at {path}")
            print("Please ensure the 'data' directory contains the required files.")
            return

    try:
        loaded_data = load_hydrogeological_data(**data_paths)
    except Exception as e:
        print(f"ERROR LOADING DATA: {e}")
        return

    # Initialize the calibrator in 'filtering_only' mode for a quick test
    run_params = {
        'run_name': 'example_test_run',
        'operation_mode': 'filtering_only',
        'generate_event_plots': False # Set to True if you want to see the plots
    }

    calibrator = AdvancedHydrogeologicalCalibrator(**run_params)
    calibrator.set_data(loaded_data)
    
    print("\nStarting the extraction and filtering pipeline...")
    calibrator.execute()

    print("\n====== QUICK TEST EXAMPLE FINISHED ======")
    print(f"Results are available in the 'results/example_test_run' folder.")

if __name__ == '__main__':
    run_example()
