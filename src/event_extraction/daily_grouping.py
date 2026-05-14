import pandas as pd

def extract_potential_daily_windows(
    observed_df: pd.DataFrame,
    min_points_day: int = 12,
    min_recovery_points: int = 4
) -> list:
    """
    Identifies and extracts potential recovery windows (from minimum to maximum daily level)
    without performing any regression analysis.
    """
    print("\n--- Identifying Potential Recovery Windows (Daily) ---")
    windows = []
    wells_with_data = observed_df['well_id'].unique()

    for i, well_id in enumerate(wells_with_data):
        print(f"  Processing well {well_id} ({i+1}/{len(wells_with_data)})...", end='\r')
        well_df = observed_df[observed_df['well_id'] == well_id]

        for date, day_df in well_df.groupby(pd.Grouper(key='datetime', freq='D')):
            if len(day_df) < min_points_day:
                continue

            idx_min = day_df['head_m'].idxmin()
            idx_max = day_df['head_m'].idxmax()

            if idx_min >= idx_max:
                continue

            potential_window_df = day_df.loc[idx_min:idx_max]
            
            if len(potential_window_df) >= min_recovery_points:
                windows.append({
                    'well_id': well_id,
                    'date': date.date(),
                    'window_df': potential_window_df,
                    'complete_group_df': day_df
                })

    print(f"\n-> {len(windows)} potential recovery windows identified.")
    return windows
