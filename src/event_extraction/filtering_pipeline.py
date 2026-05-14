import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import List, Dict, Callable, Tuple, Optional
import os
import json

from src.utils.save_filtered_data import save_discarded_events

# --- Pipeline Stage Functions ---

def stage_1_group_by_night(
    observed_df: pd.DataFrame, 
    min_night_points: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stage 1: Groups data by night and creates initial nightly windows.
    Discards nights with too few data points.
    """
    nightly_windows = []
    discards = []
    wells_with_data = observed_df['well_id'].unique()

    df = observed_df.copy()
    df['group_time'] = df['datetime'] - pd.Timedelta(hours=18)

    for well_id in wells_with_data:
        well_df = df[df['well_id'] == well_id]
        for group_date, night_df in well_df.groupby(pd.Grouper(key='group_time', freq='D')):
            if len(night_df) < min_night_points:
                discards.append({
                    'well_id': well_id,
                    'date': group_date.date(),
                    'reason': 'Not enough points in the night',
                    'n_points': len(night_df)
                })
            else:
                nightly_windows.append({
                    'well_id': well_id,
                    'date': group_date.date(),
                    'complete_group_df': night_df
                })
    return nightly_windows, discards

def stage_2_define_prw(
    nightly_windows: List[Dict], 
    min_recovery_points: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stage 2: Defines 'Potential Recovery Windows' (PRW) from nightly windows.
    Filter: Maximum level must occur after minimum level.
    """
    prw_windows = []
    discards = []
    for window in nightly_windows:
        night_df = window['complete_group_df']
        idx_min = night_df['head_m'].idxmin()
        idx_max = night_df['head_m'].idxmax()

        if idx_min >= idx_max:
            discards.append({
                **window, 
                'reason': 'Maximum level does not occur after minimum'
            })
            continue

        potential_window_df = night_df.loc[idx_min:idx_max]
        
        if len(potential_window_df) < min_recovery_points:
            discards.append({
                **window, 
                'reason': 'Potential recovery window does not have enough points',
                'n_rec_points': len(potential_window_df)
            })
        else:
            window['window_df'] = potential_window_df
            prw_windows.append(window)
            
    return prw_windows, discards

def stage_3_filter_by_reported_drawdown(
    prw_windows: List[Dict], 
    get_well_info_func: Callable, 
    min_drawdown_ratio: float
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stage 3: Filters PRW if its magnitude does not reach a minimum of the reported drawdown.
    Filter: window_delta_h >= min_drawdown_ratio * reported_drawdown
    """
    filtered_windows = []
    discards = []
    for window in prw_windows:
        well_id = window['well_id']
        well_info = get_well_info_func(well_id, 'public')
        
        if well_info.empty or 'static_level_m' not in well_info.columns or 'dynamic_level_m' not in well_info.columns or \
           well_info['static_level_m'].isna().all() or well_info['dynamic_level_m'].isna().all():
            discards.append({**window, 'reason': 'Missing Static/Dynamic data for the well'})
            continue
        
        reported_drawdown = well_info.iloc[0]['dynamic_level_m'] - well_info.iloc[0]['static_level_m']
        if reported_drawdown <= 0:
            discards.append({**window, 'reason': 'Reported drawdown is zero or negative'})
            continue

        window_df = window['window_df']
        window_magnitude = window_df['head_m'].iloc[-1] - window_df['head_m'].iloc[0]
        
        if window_magnitude < (min_drawdown_ratio * reported_drawdown):
            discards.append({
                **window, 
                'reason': 'Potential window magnitude is insufficient',
                'window_magnitude': window_magnitude,
                'required_reported_drawdown': min_drawdown_ratio * reported_drawdown
            })
        else:
            window['reported_drawdown'] = reported_drawdown
            filtered_windows.append(window)
            
    return filtered_windows, discards

def stage_4_find_candidates_with_fit(
    filtered_prw_windows: List[Dict], 
    min_recovery_points: int, 
    min_r_squared_selection: float, 
    max_delta_t_minutes: Optional[float]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stage 4: Finds ALL candidate sub-windows that meet regression criteria.
    Filter: R² > min_r_squared, slope > 0, and uniform time steps.
    """
    events_with_candidates = []
    discards = []
    for window in filtered_prw_windows:
        window_df = window['window_df']
        candidate_events = []
        
        for k in range(len(window_df) - min_recovery_points + 1):
            sub_window_df = window_df.iloc[k:]

            time_diffs = sub_window_df['datetime'].diff().dt.total_seconds() / 60.0
            if max_delta_t_minutes is not None and not time_diffs.iloc[1:].le(max_delta_t_minutes).all():
                continue
            
            sub_window_df = sub_window_df.copy()
            sub_window_df['t_min'] = (sub_window_df['datetime'] - sub_window_df['datetime'].iloc[0]).dt.total_seconds() / 60.0
            
            valid_points = sub_window_df[sub_window_df['t_min'] > 0]
            if len(valid_points) < 2: continue

            try:
                slope, _, r_value, _, _ = linregress(np.log10(valid_points['t_min']), valid_points['head_m'])
                r2 = r_value**2
                if not np.isnan(slope) and slope > 0 and r2 >= min_r_squared_selection:
                    event_magnitude = sub_window_df['head_m'].iloc[-1] - sub_window_df['head_m'].iloc[0]
                    candidate_events.append({
                        'r2': r2, 'slope': slope, 
                        'event_df': sub_window_df, 'magnitude': event_magnitude
                    })
            except (ValueError, TypeError):
                continue
        
        if not candidate_events:
            discards.append({**window, 'reason': 'No sub-windows found with valid fit (R², slope, etc.)'})
            continue

        # Attach all candidates to the event
        updated_event = {**window, 'candidates': candidate_events}
        events_with_candidates.append(updated_event)

    return events_with_candidates, discards

def stage_5_select_best_candidate(
    events_with_candidates: List[Dict], 
    min_drawdown_ratio: float
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stage 5: From the list of candidates with good R², selects the best one meeting magnitude condition.
    Filter: event_delta_h >= min_drawdown_ratio * reported_drawdown
    Selection: The candidate that passes the filter and has the highest R².
    """
    selected_events = []
    discards = []
    for event in events_with_candidates:
        candidates = event['candidates']
        required_drawdown = event['reported_drawdown'] * min_drawdown_ratio

        # Filter candidates by magnitude
        valid_candidates = [c for c in candidates if c['magnitude'] >= required_drawdown]

        if not valid_candidates:
            discards.append({
                **event, 
                'reason': 'No candidate with good R² met the magnitude requirement'
            })
            continue

        # Of those that comply, choose the one with the best R²
        best_candidate = sorted(valid_candidates, key=lambda x: x['r2'], reverse=True)[0]
        
        final_event = {**event, 'best_fit': best_candidate}
        selected_events.append(final_event)
            
    return selected_events, discards

def stage_6_filter_by_fading_recovery(
    magnitude_filtered_events: List[Dict]
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Stage 6: Ensures initial recovery rate is decreasing ("Fading recovery").
    """
    final_events_list = []
    discards = []
    for event in magnitude_filtered_events:
        event_df = event['best_fit']['event_df']
        
        if len(event_df) < 3:
            # Cannot verify, assume valid
            pass
        else:
            level_0 = event_df.iloc[0]['head_m']
            level_1 = event_df.iloc[1]['head_m']
            level_2 = event_df.iloc[2]['head_m']

            recovery_interval_1 = level_1 - level_0
            recovery_interval_2 = level_2 - level_1

            if not (recovery_interval_1 > recovery_interval_2):
                discards.append({
                    **event,
                    'reason': 'Recovery is not strictly decreasing in the first 3 points',
                    'recovery_1': recovery_interval_1,
                    'recovery_2': recovery_interval_2
                })
                continue
        
        # Build final event for DataFrame
        best_event_df = event['best_fit']['event_df']
        duration = best_event_df['datetime'].max() - best_event_df['datetime'].min()
        
        final_events_list.append({
            'well_id': event['well_id'],
            'date': event['date'],
            'slope': event['best_fit']['slope'],
            'r_squared': event['best_fit']['r2'],
            'recovery_duration_hr': duration.total_seconds() / 3600,
            'event_data': {
                'complete_group_df': event['complete_group_df'], 
                'event_df': best_event_df
            }
        })

    return pd.DataFrame(final_events_list), discards

def stage_7_filter_slope_outliers(
    final_events: pd.DataFrame, 
    min_events_per_well_for_iqr: int = 5
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Stage 7 (Optional): Filters events whose slope is a statistical outlier (IQR).
    """
    if final_events.empty:
        return final_events, []

    indices_to_keep = []
    discards = []
    grouped_wells = final_events.groupby('well_id')
    
    for well_id, group in grouped_wells:
        if len(group) >= min_events_per_well_for_iqr:
            Q1 = group['slope'].quantile(0.25)
            Q3 = group['slope'].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            outliers_mask = (group['slope'] < lower_limit) | (group['slope'] > upper_limit)
            outliers = group[outliers_mask]
            
            for _, outlier in outliers.iterrows():
                discards.append({
                    'well_id': outlier['well_id'],
                    'date': outlier['date'],
                    'reason': 'Slope outside IQR range (outlier)',
                    'slope': outlier['slope'],
                    'lower_limit': lower_limit,
                    'upper_limit': upper_limit,
                    'Q1': Q1, 'Q3': Q3
                })
            
            indices_to_keep.extend(group[~outliers_mask].index)
        else:
            indices_to_keep.extend(group.index)
    
    filtered_events = final_events.loc[indices_to_keep]
    return filtered_events, discards

# --- Helper function for analysis ---

def _analyze_stage(events: List[Dict]) -> Dict:
    """
    Analyzes a list of events to count unique wells and events per well.
    """
    if not events:
        return {
            'num_events': 0,
            'num_unique_wells': 0,
            'wells_events': {},
            'wells_list': []
        }

    df_temp = pd.DataFrame(events)
    events_per_well = df_temp.groupby('well_id').size().to_dict()
    unique_wells = df_temp['well_id'].unique().tolist()

    return {
        'num_events': len(events),
        'num_unique_wells': len(unique_wells),
        'wells_events': events_per_well,
        'wells_list': unique_wells
    }

# --- Pipeline Orchestrator ---

def execute_filtering_pipeline(
    observed_df: pd.DataFrame,
    get_well_info_func: Callable,
    results_dir: str,
    min_night_points: int,
    min_recovery_points: int,
    min_drawdown_ratio: float,
    min_r_squared_selection: float,
    max_delta_t_minutes: Optional[float],
    filter_slope_outliers: bool,
    min_events_per_well_for_iqr: int
) -> pd.DataFrame:
    """
    Orchestrates the execution of the entire filtering pipeline, saving results
    and detailed analysis of each stage.
    """
    print("\n--- Executing New Filtering Pipeline by Stages ---")
    analysis_summary = {}

    # --- Stage 1: Group by Night ---
    nightly_windows, discards_s1 = stage_1_group_by_night(observed_df, min_night_points)
    save_discarded_events(discards_s1, results_dir, "discards_stage_1_insufficient_points.csv")
    analysis_summary['Stage 1: Nightly Windows'] = _analyze_stage(nightly_windows)
    print(f"Stage 1: {len(nightly_windows)} initial nightly windows. ({len(discards_s1)} discarded)")

    # --- Stage 2: Define PRW ---
    prw_windows, discards_s2 = stage_2_define_prw(nightly_windows, min_recovery_points)
    save_discarded_events(discards_s2, results_dir, "discards_stage_2_no_recovery.csv", drop_dfs=True)
    analysis_summary['Stage 2: PRWs Defined'] = _analyze_stage(prw_windows)
    print(f"Stage 2: {len(prw_windows)} potential recovery windows (PRW). ({len(discards_s2)} discarded)")

    # --- Stage 3: Filter by Reported Drawdown ---
    filtered_prw, discards_s3 = stage_3_filter_by_reported_drawdown(prw_windows, get_well_info_func, min_drawdown_ratio)
    save_discarded_events(discards_s3, results_dir, "discards_stage_3_insufficient_window_magnitude.csv", drop_dfs=True)
    analysis_summary['Stage 3: PRWs by Magnitude'] = _analyze_stage(filtered_prw)
    print(f"Stage 3: {len(filtered_prw)} PRWs post-magnitude filter. ({len(discards_s3)} discarded)")

    # --- Stage 4: Find Candidates with Fit ---
    events_with_candidates, discards_s4 = stage_4_find_candidates_with_fit(filtered_prw, min_recovery_points, min_r_squared_selection, max_delta_t_minutes)
    save_discarded_events(discards_s4, results_dir, "discards_stage_4_no_valid_candidates.csv", drop_dfs=True)
    analysis_summary['Stage 4: Events with Candidates'] = _analyze_stage(events_with_candidates)
    print(f"Stage 4: {len(events_with_candidates)} events with at least one valid fit candidate. ({len(discards_s4)} discarded)")

    # --- Stage 5: Select Best Candidate by Magnitude and R² ---
    selected_events, discards_s5 = stage_5_select_best_candidate(events_with_candidates, min_drawdown_ratio)
    save_discarded_events(discards_s5, results_dir, "discards_stage_5_no_candidate_met_magnitude.csv", drop_dfs=True)
    analysis_summary['Stage 5: Events with Best Candidate Selected'] = _analyze_stage(selected_events)
    print(f"Stage 5: {len(selected_events)} events with a final candidate selected. ({len(discards_s5)} discarded)")

    # --- Stage 6: Filter by Fading Recovery ---
    almost_final_events_df, discards_s6 = stage_6_filter_by_fading_recovery(selected_events)
    save_discarded_events(discards_s6, results_dir, "discards_stage_6_non_fading_recovery.csv", drop_dfs=True)
    analysis_summary['Stage 6: Fading Recovery'] = _analyze_stage(almost_final_events_df.to_dict('records'))
    print(f"Stage 6: {len(almost_final_events_df)} events with fading recovery. ({len(discards_s6)} discarded)")

    # --- Stage 7: Filter Slope Outliers ---
    if filter_slope_outliers:
        final_events, discards_s7 = stage_7_filter_slope_outliers(almost_final_events_df, min_events_per_well_for_iqr)
        save_discarded_events(discards_s7, results_dir, "discards_stage_7_slope_outliers.csv")
        analysis_summary['Stage 7: No Slope Outliers'] = _analyze_stage(final_events.to_dict('records'))
        print(f"Stage 7: {len(final_events)} final events post-outlier filter. ({len(discards_s7)} discarded)")
    else:
        final_events = almost_final_events_df
        analysis_summary['Stage 7: No Slope Outliers'] = analysis_summary['Stage 6: Fading Recovery']
        print("Stage 7: Slope outlier filter skipped.")

    # Save analysis summary
    summary_path = os.path.join(results_dir, 'filtering_summary_by_stage.json')
    with open(summary_path, 'w') as f:
        json.dump(analysis_summary, f, indent=4)
    print(f"\n-> Detailed filtering analysis saved in: {summary_path}")

    print(f"\n-> Pipeline finished. {len(final_events)} final events selected.")
    return final_events
