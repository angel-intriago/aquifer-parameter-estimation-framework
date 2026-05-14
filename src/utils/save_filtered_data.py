import pandas as pd
import os
from typing import List, Dict, Any

def save_potential_windows(windows: List[Dict[str, Any]], results_dir: str, subdirectory: str, base_name: str):
    """
    Saves a list of potential windows into individual CSV files.
    """
    output_dir = os.path.join(results_dir, 'filtered_data', subdirectory)
    os.makedirs(output_dir, exist_ok=True)

    if not windows:
        return

    summary_data = []
    for i, window in enumerate(windows):
        well_id = window['well_id']
        date = window['date']
        df = window['window_df']
        
        filename = f"{base_name}_{well_id}_{date}_{i}.csv"
        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, index=False)
        
        summary_data.append({
            'well_id': well_id,
            'date': date,
            'num_points': len(df),
            'file': filename
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f'_summary_{base_name}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"-> {len(windows)} windows/events saved in: {output_dir}")

def save_discarded_events(discarded_events: List[Dict[str, Any]], results_dir: str, filename: str, drop_dfs: bool = False):
    """
    Saves a list of discarded events and the reason into a single CSV file.
    If `drop_dfs` is True, it removes any nested DataFrames before saving.
    """
    if not discarded_events:
        return

    output_dir = os.path.join(results_dir, 'filtered_data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a copy to avoid modifying the original list
    events_to_save = []
    for event in discarded_events:
        event_copy = event.copy()
        if drop_dfs:
            # Remove nested DataFrames that cannot be directly serialized to CSV
            keys_to_remove = [k for k, v in event_copy.items() if isinstance(v, pd.DataFrame)]
            for k in keys_to_remove:
                del event_copy[k]
        events_to_save.append(event_copy)

    discarded_df = pd.DataFrame(events_to_save)
    file_path = os.path.join(output_dir, filename)
    discarded_df.to_csv(file_path, index=False)
    print(f"-> {len(discarded_events)} discarded event records saved in: {file_path}")
