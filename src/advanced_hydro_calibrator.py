import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import linregress
from scipy.interpolate import Rbf
import warnings
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

from src.data_loading import load_hydrogeological_data
from src.event_extraction.daily_grouping import extract_potential_daily_windows
from src.event_extraction.filtering_pipeline import execute_filtering_pipeline
from src.calibration.differential_evolution import execute_de_optimization
from src.visualization import (
    plot_scatter_Q, plot_boxplot_K, plot_map_T, plot_recovery_event, 
    plot_correlation_heatmap, plot_error_map, plot_neighborhood_drawdown, 
    plot_comparative_flow, plot_iteration_history
)
from src.utils.save_filtered_data import save_potential_windows, save_discarded_events


class AdvancedHydrogeologicalCalibrator:
    """
    Calibrates hydrogeological parameters (K, S) and pumping flow rates (Q)
    using nightly recovery curve analysis, considering variable saturated 
    thickness and neighbor well interference.
    """

    def __init__(self,
                 # Run parameters
                 run_name: Optional[str] = None,
                 operation_mode: str = 'complete', # 'complete' or 'filtering_only'
                 
                 # Methods and visualization parameters
                 calibration_method: str = 'differential_evolution',
                 consider_interference: bool = False,
                  generate_event_plots: bool = False,
                 generate_summary_plots: bool = True,

                 # Search range parameters
                 conductivity_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                 storage_range: Tuple[float, float] = (0.0001, 0.01),
                 
                 # Flow rate calibration control parameters
                 calibrate_target_well_flow: bool = True,
                 calibrate_public_well_flows: bool = True,
                 calibrate_private_well_flows: bool = True,
                 target_uncertainty: Tuple[float, float] = (0.9, 1.1),
                 public_uncertainty: Tuple[float, float] = (0.9, 1.1),
                 private_uncertainty: Tuple[float, float] = (0.5, 2.0),
                 neighborhood_uncertainty: Optional[Dict[str, Union[Tuple[float, float], Dict]]] = None,
                 
                 # Physical model and neighborhood parameters
                 search_radius_m: float = 2000.0,
                 rbf_interpolator_function: str = 'thin_plate',
                 rbf_interpolator_smooth: float = 0.0,
                 default_T_if_no_strat: float = 100.0,
                 default_K_if_no_formation: float = 1.0,
                 min_calculated_T: float = 1e-6,

                 # Data filtering parameters
                 min_extraction_points: int = 12,
                 min_recovery_points: int = 4,
                 min_interpolator_wells: int = 4,
                 min_events_per_well: int = 5,
                 min_events_per_well_for_iqr: int = 5,
                 filter_slope_outliers: bool = False,
                 min_r_squared_selection: float = 0.85,
                 min_drawdown_ratio: float = 0.5,
                 max_delta_t_minutes: float = 15.0,

                 # Optimizer control parameters
                 maxiter: int = 300,
                 popsize: int = 25,
                 seed: int = 42,
                 polish: bool = True,
                 updating: str = 'deferred',
                 workers: int = 1,

                 # Iterative consistency calibration parameters
                 consistency_iterations: int = 1):
        
        # 1. Parameter assignment to class attributes
        self.run_name = run_name
        self.operation_mode = operation_mode
        self.calibration_method = calibration_method
        self.consider_interference = consider_interference
        self.generate_event_plots = generate_event_plots
        self.generate_summary_plots = generate_summary_plots
        self.conductivity_ranges = conductivity_ranges
        self.storage_range = storage_range
        self.calibrate_target_well_flow = calibrate_target_well_flow
        self.calibrate_public_well_flows = calibrate_public_well_flows
        self.calibrate_private_well_flows = calibrate_private_well_flows
        self.target_uncertainty = target_uncertainty
        self.public_uncertainty = public_uncertainty
        self.private_uncertainty = private_uncertainty
        self.neighborhood_uncertainty = neighborhood_uncertainty
        self.search_radius_m = search_radius_m
        self.rbf_interpolator_function = rbf_interpolator_function
        self.rbf_interpolator_smooth = rbf_interpolator_smooth
        self.default_T_if_no_strat = default_T_if_no_strat
        self.default_K_if_no_formation = default_K_if_no_formation
        self.min_calculated_T = min_calculated_T
        self.min_extraction_points = min_extraction_points
        self.min_recovery_points = min_recovery_points
        self.min_interpolator_wells = min_interpolator_wells
        self.min_events_per_well = min_events_per_well
        self.min_events_per_well_for_iqr = min_events_per_well_for_iqr
        self.filter_slope_outliers = filter_slope_outliers
        self.min_r_squared_selection = min_r_squared_selection
        self.min_drawdown_ratio = min_drawdown_ratio
        self.max_delta_t_minutes = max_delta_t_minutes
        self.maxiter = maxiter
        self.popsize = popsize
        self.seed = seed
        self.polish = polish
        self.updating = updating
        self.workers = workers
        self.consistency_iterations = consistency_iterations

        # Default conductivity ranges if not provided
        if self.conductivity_ranges is None:
            self.conductivity_ranges = {
                'Fm Serra Geral': (1e-5, 1e-3), 'Fm Botucatu': (0.1, 15.0),
                'Fm Guará Sup': (0.5, 5.0), 'Fm Guará Inf': (0.2, 3.0),
                'Fm Piramboia Sup': (0.01, 2.0), 'Fm Piramboia Inf': (0.01, 2.0),
                'Fm Diabásio': (1e-5, 1e-3), 'Fm Corumbataí': (0.01, 0.5)
            }

        # 2. Results directory construction
        if self.run_name:
            subfolder_name = self.run_name
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            subfolder_name = f"{timestamp}_{self.calibration_method}"
            
        self.results_dir = os.path.join('results', subfolder_name)
        self.viz_data_dir = os.path.join(self.results_dir, 'visualization_data')
        self.filtered_data_dir = os.path.join(self.results_dir, 'filtered_data')
        self.event_data_dir = os.path.join(self.viz_data_dir, 'recovery_events')
        self.neighborhood_data_dir = os.path.join(self.viz_data_dir, 'neighborhood_analysis')
        self.summary_data_dir = os.path.join(self.viz_data_dir, 'global_summary')

        # 3. State variable initialization
        self.observed_df = None
        self.private_wells = None
        self.public_wells = None
        self.private_stratigraphy = None
        self.public_stratigraphy = None
        self.daily_events = pd.DataFrame()
        self.level_interpolator = None
        self.iteration_summary_history = []

        # 4. Environment setup
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.event_data_dir, exist_ok=True)
        os.makedirs(self.neighborhood_data_dir, exist_ok=True)
        os.makedirs(self.summary_data_dir, exist_ok=True)
        os.makedirs(self.filtered_data_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning, module='geopandas')
        print(f"Results directory: {os.path.abspath(self.results_dir)}")

    def set_data(self, data: Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]]):
        """Assigns loaded dataframes to the class instance."""
        self.observed_df = data["observed_df"]
        self.private_wells = data["private_wells"]
        self.public_wells = data["public_wells"]
        self.private_stratigraphy = data["private_stratigraphy"]
        self.public_stratigraphy = data["public_stratigraphy"]
        print("-> Data assigned to calibrator.")

    def _create_level_interpolator(self):
        print("\n--- 2a. Creating Spatial Interpolator for Dynamic Levels ---")
        wells = self.public_wells
        valid_wells = wells[wells['ground_elev_m'].notna() & wells['dynamic_level_m'].notna()].copy()
        
        # Remove wells with duplicate coordinates to avoid singular matrix in Rbf
        if not valid_wells.empty:
            num_before = len(valid_wells)
            wkt_geometries = valid_wells.geometry.to_wkt()
            valid_wells = valid_wells.loc[wkt_geometries.drop_duplicates().index]
            num_after = len(valid_wells)
            if num_before > num_after:
                print(f"WARNING: Removed {num_before - num_after} wells with duplicate coordinates.")

        if len(valid_wells) < 4:
            print("WARNING: Not enough unique public wells with Elevation and Dynamic Level data to create a robust interpolator.")
            self.level_interpolator = None
            return

        valid_wells['water_level_m'] = valid_wells['ground_elev_m'] - valid_wells['dynamic_level_m']
        
        x = valid_wells.geometry.x.values
        y = valid_wells.geometry.y.values
        z = valid_wells['water_level_m'].values
        
        self.level_interpolator = Rbf(x, y, z, function='thin_plate', smooth=0)
        print("-> Interpolator created successfully.")

    def extract_recovery_events(self):
        """
        Executes the staged filtering pipeline to extract final recovery events.
        """
        self.daily_events = execute_filtering_pipeline(
            observed_df=self.observed_df,
            get_well_info_func=self._get_well_info,
            results_dir=self.results_dir,
            min_night_points=self.min_extraction_points,
            min_recovery_points=self.min_recovery_points,
            min_drawdown_ratio=self.min_drawdown_ratio,
            min_r_squared_selection=self.min_r_squared_selection,
            max_delta_t_minutes=self.max_delta_t_minutes,
            filter_slope_outliers=self.filter_slope_outliers,
            min_events_per_well_for_iqr=self.min_events_per_well_for_iqr
        )

        if not self.daily_events.empty:
            if self.generate_event_plots:
                wells_with_events = self.daily_events['well_id'].unique()
                for well_id in wells_with_events:
                    well_events = self.daily_events[self.daily_events['well_id'] == well_id]
                    self._generate_event_plots(well_events)

            events_for_csv = self.daily_events.drop(columns=['event_data'])
            events_path = os.path.join(self.results_dir, 'extracted_recovery_events.csv')
            events_for_csv.to_csv(events_path, index=False)
            print(f"   Intermediate results (without plot data) saved in: {events_path}")

    def _prepare_neighborhood(self, well_id, neighborhood_info_cache):
        if well_id in neighborhood_info_cache:
            return neighborhood_info_cache[well_id]

        target_info = self._get_well_info(well_id, 'public')
        if target_info.empty or 'ground_elev_m' not in target_info.columns or target_info['ground_elev_m'].isna().all() or 'dynamic_level_m' not in target_info.columns or target_info['dynamic_level_m'].isna().all():
            print(f"  Skipping well {well_id}: missing Elevation or Dynamic Level data.")
            return None, None, None
        
        target_coord = target_info.geometry.iloc[0]
        central_water_level = target_info.iloc[0]['ground_elev_m'] - target_info.iloc[0]['dynamic_level_m']

        neighborhood_info = {well_id: {'type': 'public', 'water_level_m': central_water_level, 'Q_m3d': target_info.iloc[0]['Q_m3d'], 'distance': 0}}
        local_formations = set(self._get_stratigraphy(well_id, 'public')['descricao'].unique())

        all_wells = pd.concat([
            self.public_wells.rename(columns={'well_id': 'unified_id'})[['unified_id', 'Q_m3d', 'ground_elev_m', 'dynamic_level_m', 'geometry']].assign(type='public'),
            self.private_wells.rename(columns={'well_id': 'unified_id'})[['unified_id', 'Q_m3d', 'ground_elev_m', 'geometry']].assign(type='private')
        ], ignore_index=True)
        all_wells = all_wells[all_wells['unified_id'] != well_id]
        all_wells['distance'] = all_wells.geometry.apply(lambda g: target_coord.distance(g))
        neighbors = all_wells[all_wells['distance'] <= self.search_radius_m].copy()
        
        for _, neighbor in neighbors.iterrows():
            neighbor_id, neighbor_type = neighbor['unified_id'], neighbor['type']
            neighbor_level = np.nan
            if neighbor_type == 'public' and pd.notna(neighbor['ground_elev_m']) and pd.notna(neighbor['dynamic_level_m']):
                neighbor_level = neighbor['ground_elev_m'] - neighbor['dynamic_level_m']
            elif self.level_interpolator:
                neighbor_level = self.level_interpolator(neighbor.geometry.x, neighbor.geometry.y)
            
            if pd.notna(neighbor_level):
                neighborhood_info[neighbor_id] = {'type': neighbor_type, 'water_level_m': float(neighbor_level), 'Q_m3d': neighbor['Q_m3d'], 'distance': neighbor['distance']}
                local_formations.update(self._get_stratigraphy(neighbor_id, neighbor_type)['descricao'].unique())
        
        neighborhood_info_cache[well_id] = (neighborhood_info, local_formations, target_info)
        return neighborhood_info, local_formations, target_info

    def _configure_optimization(self, well_id, neighborhood_info, local_formations, iter_num, global_calibrated_flows):
        param_names = []
        bounds = []
        fixed_params = {}

        general_neighborhood_uncertainty_range = None
        detailed_neighborhood_uncertainty_config = None

        if self.neighborhood_uncertainty and well_id in self.neighborhood_uncertainty:
            neighborhood_config = self.neighborhood_uncertainty[well_id]
            if isinstance(neighborhood_config, dict):
                detailed_neighborhood_uncertainty_config = neighborhood_config
                print(f"  -> Using DETAILED uncertainty configuration for neighborhood '{well_id}'.")
            elif isinstance(neighborhood_config, tuple):
                general_neighborhood_uncertainty_range = neighborhood_config
                print(f"  -> Using GENERAL uncertainty range for neighborhood '{well_id}': {general_neighborhood_uncertainty_range}")

        for member_id, member_info in neighborhood_info.items():
            q_base_for_bounds = member_info.get('Q_m3d', 0)
            q_value_for_param = global_calibrated_flows.get(member_id, q_base_for_bounds)
            
            calibrate_this_q = False
            
            if general_neighborhood_uncertainty_range:
                uncertainty_range = general_neighborhood_uncertainty_range
            elif detailed_neighborhood_uncertainty_config:
                if member_id == well_id:
                    uncertainty_range = detailed_neighborhood_uncertainty_config.get('target', self.target_uncertainty)
                elif member_info['type'] == 'public':
                    uncertainty_range = detailed_neighborhood_uncertainty_config.get('publics', self.public_uncertainty)
                elif member_info['type'] == 'private':
                    uncertainty_range = detailed_neighborhood_uncertainty_config.get('privates', self.private_uncertainty)
                else:
                    uncertainty_range = (1.0, 1.0)
            else:
                if member_id == well_id:
                    uncertainty_range = self.target_uncertainty
                elif member_info['type'] == 'public':
                    uncertainty_range = self.public_uncertainty
                elif member_info['type'] == 'private':
                    uncertainty_range = self.private_uncertainty
                else:
                    uncertainty_range = (1.0, 1.0)

            if member_id == well_id:
                calibrate_this_q = self.calibrate_target_well_flow
            elif member_info['type'] == 'public':
                calibrate_this_q = self.calibrate_public_well_flows
            elif member_info['type'] == 'private':
                calibrate_this_q = self.calibrate_private_well_flows
            
            q_min = q_base_for_bounds * uncertainty_range[0]
            q_max = q_base_for_bounds * uncertainty_range[1]

            if iter_num > 0 and hasattr(self, 'consistent_parameters') and well_id in self.consistent_parameters:
                consistent_group_info = self.consistent_parameters[well_id]
                if well_id != consistent_group_info['reference_well_id']:
                    q_consistent_key = f"Q_{member_id}"
                    if q_consistent_key in consistent_group_info:
                        q_value_for_param = consistent_group_info[q_consistent_key]
                        if pd.isna(q_value_for_param):
                            q_value_for_param = q_base_for_bounds
                        calibrate_this_q = False
                        q_value_for_param = np.clip(q_value_for_param, q_min, q_max)

            if calibrate_this_q:
                param_names.append(f"Q_{member_id}")
                bounds.append((q_min, q_max))
            else:
                fixed_q_clipped = np.clip(q_value_for_param, q_min, q_max)
                fixed_params[f"Q_{member_id}"] = fixed_q_clipped

        if iter_num > 0 and hasattr(self, 'consistent_parameters') and well_id in self.consistent_parameters:
            best_params = self.consistent_parameters[well_id]
            if well_id != best_params['reference_well_id']:
                print("  -> Using consistent K and S from the best neighborhood. Optimizing Q.")
                fixed_ks_params = {k: v for k, v in best_params.items() if k.startswith('K_') or k == 'S'}
                fixed_params.update(fixed_ks_params)
            else:
                print("  -> This is the best neighborhood. Refining all parameters.")
        
        if 'S' not in fixed_params:
            param_names.append("S")
            bounds.append(self.storage_range)
        
        for formation in sorted(list(local_formations)):
            if formation == 'Fm Serra Geral':
                continue

            k_param_name = f"K_{formation}"
            if k_param_name not in fixed_params and formation in self.conductivity_ranges:
                param_names.append(k_param_name)
                bounds.append(self.conductivity_ranges[formation])
        
        return param_names, bounds, fixed_params if fixed_params else None

    def _analyze_iteration_consistency(self, iter_results_df, neighborhood_info_cache):
        print("\n--- Stage 2: Analyzing parameter consistency between neighborhoods ---")
        
        well_to_neighborhoods = {}
        for calibrated_well_id in iter_results_df['calibrated_well_id']:
            neighborhood_info, _, _ = neighborhood_info_cache[calibrated_well_id]
            for member_id in neighborhood_info.keys():
                if member_id not in well_to_neighborhoods:
                    well_to_neighborhoods[member_id] = []
                well_to_neighborhoods[member_id].append(calibrated_well_id)

        connected_groups = []
        visited = set()
        for calibrated_well_id in iter_results_df['calibrated_well_id']:
            if calibrated_well_id not in visited:
                current_component = set()
                stack = [calibrated_well_id]
                while stack:
                    current_well = stack.pop()
                    if current_well not in visited:
                        visited.add(current_well)
                        current_component.add(current_well)
                        for member_neighborhood_id in well_to_neighborhoods.get(current_well, []):
                            stack.append(member_neighborhood_id)
                        for parent_neighborhood_id in well_to_neighborhoods.get(current_well, []):
                            stack.append(parent_neighborhood_id)
                connected_groups.append(list(current_component))
        
        print(f"  Found {len(connected_groups)} interconnected neighborhood groups.")

        consistent_parameters = {}
        for group in connected_groups:
            group_df = iter_results_df[iter_results_df['calibrated_well_id'].isin(group)]
            if group_df.empty: continue
            
            best_result = group_df.loc[group_df['final_error'].idxmin()]
            best_well_id = best_result['calibrated_well_id']
            best_params = best_result.to_dict()
            best_params['reference_well_id'] = best_well_id

            print(f"  -> For group {group}, the best neighborhood is '{best_well_id}' (Error: {best_result['final_error']:.4f}).")

            for member_well_id in group:
                consistent_parameters[member_well_id] = best_params
        
        return consistent_parameters

    def _log_iteration_summary(self, iter_results_df, iter_num):
        iteration_summary = {'iter_num': iter_num + 1}
        iteration_summary['average_error'] = iter_results_df['final_error'].mean()
        
        s_vals_iter = iter_results_df['S'].dropna().tolist()
        if s_vals_iter:
            iteration_summary['S_median'] = np.median(s_vals_iter)
        
        for col in iter_results_df.columns:
            if col.startswith('K_'):
                formation = col.replace('K_', '')
                k_vals_iter = iter_results_df[col].dropna().tolist()
                if k_vals_iter:
                    iteration_summary[f'K_{formation}_median'] = np.median(k_vals_iter)
        
        self.iteration_summary_history.append(iteration_summary)
        print(f"  -> Summary for iteration {iter_num + 1} saved for evolution analysis.")

    def _update_final_consistent_results(self, iter_results_df, wells_to_calibrate):
        print("\n--- Last iteration completed. Updating final results with consistent parameters. ---")
        final_df = pd.DataFrame()
        for calibrated_well_id in wells_to_calibrate:
            if calibrated_well_id in self.consistent_parameters:
                consistent_params = self.consistent_parameters[calibrated_well_id]
                original_result = iter_results_df[iter_results_df['calibrated_well_id'] == calibrated_well_id].iloc[0].to_dict()
                for k in original_result.keys():
                    if (k.startswith('K_') or k == 'S') and k in consistent_params:
                        original_result[k] = consistent_params[k]
                final_df = pd.concat([final_df, pd.DataFrame([original_result])], ignore_index=True)
            else:
                final_df = pd.concat([final_df, iter_results_df[iter_results_df['calibrated_well_id'] == calibrated_well_id]], ignore_index=True)
        return final_df.to_dict('records')

    def _generate_event_plots(self, well_events):
        """Generates and saves plots for a list of events of a well."""
        event_plots_dir = os.path.join(self.results_dir, 'recovery_event_figures')
        os.makedirs(event_plots_dir, exist_ok=True)
        os.makedirs(self.event_data_dir, exist_ok=True)
        
        if not hasattr(self, '_printed_plotting_message'):
            print(f"\n-> Saving filtered event plots in: {event_plots_dir}")
            self._printed_plotting_message = True

        for _, event in well_events.iterrows():
            if 'event_data' in event and event['event_data']:
                plot_data = event['event_data']
                date_str = event['date'].strftime('%Y-%m-%d')
                well_id_str = str(event['well_id'])

                group_path = os.path.join(self.event_data_dir, f'event_{well_id_str}_{date_str}_group.csv')
                event_path = os.path.join(self.event_data_dir, f'event_{well_id_str}_{date_str}_event.csv')
                params_path = os.path.join(self.event_data_dir, f'event_{well_id_str}_{date_str}_params.json')

                plot_data['complete_group_df'].to_csv(group_path, index=False)
                plot_data['event_df'].to_csv(event_path, index=False)
                with open(params_path, 'w') as f:
                    json.dump({'slope': event['slope'], 'well_id': well_id_str, 'event_date': date_str}, f, indent=4)

                plot_recovery_event(
                    group_df=plot_data['complete_group_df'],
                    event_df=plot_data['event_df'],
                    slope=event['slope'],
                    well_id=event['well_id'],
                    event_date=event['date'],
                    output_dir=event_plots_dir
                )

    def _generate_final_neighborhood_plots(self, final_results_df, neighborhood_info_cache):
        print("\n--- Generating final neighborhood analysis plots ---")
        for _, res in final_results_df.iterrows():
            well_id = res['calibrated_well_id']
            param_dict = res.to_dict()
            neighborhood_info, _, target_info = neighborhood_info_cache[well_id]
            target_coord = target_info.geometry.iloc[0]
            
            all_wells = pd.concat([
                self.public_wells.rename(columns={'well_id': 'unified_id'})[['unified_id', 'geometry']].assign(type='public'),
                self.private_wells.rename(columns={'well_id': 'unified_id'})[['unified_id', 'geometry']].assign(type='private')
            ], ignore_index=True)
            all_wells['distance'] = all_wells.geometry.apply(lambda g: target_coord.distance(g))
            neighbors = all_wells[all_wells['distance'] <= self.search_radius_m].copy()

            try:
                interference_percentages = self._calculate_interference_percentages(param_dict, neighborhood_info, well_id)
                plot_neighborhood_drawdown(
                    target_coords=target_coord, neighbors_gdf=neighbors, interference_percentages=interference_percentages,
                    search_radius_m=self.search_radius_m, output_dir=self.results_dir, well_id=well_id
                )
            except Exception as e:
                print(f"  WARNING: Could not generate neighborhood plot for {well_id}. Error: {e}")

    def _get_well_info(self, well_id, type):
        df = self.public_wells if type == 'public' else self.private_wells
        return df[df['well_id'] == well_id]

    def _get_stratigraphy(self, well_id, type):
        df = self.public_stratigraphy if type == 'public' else self.private_stratigraphy
        return df[df['well_id'].astype(str) == str(well_id)]

    def _calculate_effective_transmissivity(self, well_id: str, type: str, conductivities: Dict[str, float], water_level_m: float) -> float:
        well_info = self._get_well_info(well_id, type)
        if well_info.empty or 'ground_elev_m' not in well_info.columns or well_info['ground_elev_m'].isna().all():
            return 1e-6

        well_elevation = well_info.iloc[0]['ground_elev_m']
        stratigraphy = self._get_stratigraphy(well_id, type)
        if stratigraphy.empty: return 100.0

        T = 0.0
        for _, layer in stratigraphy.iterrows():
            if layer['descricao'] == 'Fm Serra Geral':
                continue

            k = conductivities.get(layer['descricao'], 1.0)
            layer_top_elevation = well_elevation - layer['de_m']
            layer_bottom_elevation = well_elevation - layer['para_m']
            saturated_thickness = max(0, min(layer_top_elevation, water_level_m) - layer_bottom_elevation)
            T += k * saturated_thickness
            
        return max(T, 1e-6)

    def _calculate_interference_percentages(self, param_dict, neighborhood_info, target_well_id):
        calibrated_conductivities = {
            p.replace("K_", ""): v for p, v in param_dict.items() if p.startswith("K_")
        }
        s_calibrated = param_dict.get("S")
        q_calibrated_target = param_dict.get(f"Q_{target_well_id}")

        if s_calibrated is None or q_calibrated_target is None:
            return {}

        central_water_level = neighborhood_info[target_well_id]["water_level_m"]
        T_target = self._calculate_effective_transmissivity(
            target_well_id, "public", calibrated_conductivities, central_water_level
        )
        if T_target < 1e-6:
            return {}

        effects = {}
        effects[target_well_id] = q_calibrated_target

        for neighbor_id, info in neighborhood_info.items():
            if neighbor_id == target_well_id:
                continue

            T_neighbor = self._calculate_effective_transmissivity(
                neighbor_id, info["type"], calibrated_conductivities, info["water_level_m"]
            )
            distance = info["distance"]
            q_neighbor = info["Q_m3d"]

            if T_neighbor > 1e-5 and distance > 1.0:
                interference_effect = T_target * (q_neighbor / T_neighbor) * np.exp(
                    -distance ** 2 * s_calibrated / (4 * T_neighbor)
                )
                effects[neighbor_id] = interference_effect
        
        total_effect = sum(effects.values())
        
        if total_effect < 1e-9:
            return {well_id: 0.0 for well_id in effects}

        percentages = {
            well_id: (effect / total_effect) * 100
            for well_id, effect in effects.items()
        }
        
        return percentages

    def execute_complete_calibration(self):
        print("\n--- 3. Executing Iterative Advanced Calibration by Well ---")
        if self.daily_events.empty:
            print("No recovery events to calibrate. Run extraction first.")
            return

        self._create_level_interpolator()
        wells_to_calibrate = self.daily_events['well_id'].unique()

        global_calibrated_flows = {}
        neighborhood_info_cache = {}
        previous_iter_results = {}

        for iter_num in range(self.consistency_iterations):
            print(f"\n--- STARTING CONSISTENCY ITERATION {iter_num + 1}/{self.consistency_iterations} ---")
            self.calibration_results = []

            print(f"\n--- Stage 1: Executing base calibration for {len(wells_to_calibrate)} wells ---")
            for i, well_id in enumerate(wells_to_calibrate):
                print(f"\n--- Calibrating Well {well_id} ({i+1}/{len(wells_to_calibrate)}) ---")
                
                well_events = self.daily_events[self.daily_events['well_id'] == well_id]
                if len(well_events) < self.min_events_per_well:
                    print(f"  Skipping well {well_id}: insufficient events ({len(well_events)}).")
                    continue

                if iter_num == 0 and self.generate_event_plots:
                    self._generate_event_plots(well_events)

                neighborhood_info, local_formations, target_info = self._prepare_neighborhood(well_id, neighborhood_info_cache)
                if neighborhood_info is None:
                    continue

                param_names, bounds, fixed_params = self._configure_optimization(
                    well_id, neighborhood_info, local_formations, iter_num, global_calibrated_flows
                )

                args = (well_id, param_names, neighborhood_info, well_events, self._calculate_effective_transmissivity, self.consider_interference)
                
                if not param_names:
                    from src.calibration.differential_evolution import objective_function_de
                    calculated_error = objective_function_de(np.array([]), args + (fixed_params,))
                    opt_result = type('obj', (object,), {'success': True, 'fun': calculated_error, 'x': np.array([])})()
                else:
                    opt_result = execute_de_optimization(
                        bounds=bounds, args=args, maxiter=self.maxiter, popsize=self.popsize, 
                        seed=self.seed, polish=self.polish, updating=self.updating, workers=self.workers,
                        fixed_params=fixed_params
                    )

                if opt_result.success:
                    new_error = opt_result.fun
                    previous_error = previous_iter_results.get(well_id, {}).get('final_error', np.inf)

                    if iter_num == 0 or new_error < previous_error:
                        if iter_num > 0:
                            print(f"  -> Optimization completed. Error IMPROVED: {new_error:.4f} (previous: {previous_error:.4f})")
                        else:
                            print(f"  -> Optimization completed. Final error: {new_error:.4f}")

                        param_dict = dict(zip(param_names, opt_result.x))
                        if fixed_params:
                            param_dict.update(fixed_params)
                        
                        res_dict = {'calibrated_well_id': well_id, 'final_error': new_error}
                        res_dict.update(param_dict)
                        self.calibration_results.append(res_dict)

                        for p_name, p_value in param_dict.items():
                            if p_name.startswith('Q_'):
                                well_q_id = p_name.replace('Q_', '')
                                global_calibrated_flows[well_q_id] = p_value

                        comparative_q_dir = os.path.join(self.results_dir, 'comparative_q_figures', f'neighborhood_{well_id}')
                        os.makedirs(comparative_q_dir, exist_ok=True)
                        for member_id, member_info in neighborhood_info.items():
                            try:
                                reported_q = member_info.get('Q_m3d')
                                calibrated_q = param_dict.get(f'Q_{member_id}')
                                if reported_q is not None and calibrated_q is not None:
                                    plot_comparative_flow(
                                        well_id=member_id, reported_q=reported_q, calibrated_q=calibrated_q,
                                        output_dir=comparative_q_dir, custom_filename=f"q_comparison_{member_id}.png"
                                    )
                            except Exception as e:
                                print(f"  WARNING: Could not generate flow plot for {member_id}. Error: {e}")
                    else:
                        print(f"  -> WARNING: Error did not improve ({new_error:.4f} >= {previous_error:.4f}). Keeping previous parameters.")
                        old_result = previous_iter_results[well_id].copy()
                        old_result['calibrated_well_id'] = well_id
                        self.calibration_results.append(old_result)
                else:
                    print(f"  WARNING: Optimization for {well_id} did not converge.")
                    if well_id in previous_iter_results:
                        print(f"  -> Reusing previous iteration results for well {well_id}.")
                        old_result = previous_iter_results[well_id].copy()
                        old_result['calibrated_well_id'] = well_id
                        self.calibration_results.append(old_result)
            
            if not self.calibration_results:
                print("No calibration results to analyze consistency. Skipping to end.")
                break

            print("\n--- Stage 2: Analyzing parameter consistency between neighborhoods ---")
            iter_results_df = pd.DataFrame(self.calibration_results)
            
            previous_iter_results = iter_results_df.set_index('calibrated_well_id').to_dict('index')

            self.consistent_parameters = self._analyze_iteration_consistency(iter_results_df, neighborhood_info_cache)

            self._log_iteration_summary(iter_results_df, iter_num)

            if (iter_num + 1) == self.consistency_iterations:
                self.calibration_results = self._update_final_consistent_results(
                    iter_results_df, wells_to_calibrate
                )

        if self.generate_summary_plots:
            self._generate_final_neighborhood_plots(pd.DataFrame(self.calibration_results), neighborhood_info_cache)

    def save_and_aggregate_results(self):
        print("\n--- 4. Aggregating and Saving Final Results ---")
        if not self.calibration_results:
            print("No calibration results to save.")
            return

        detailed_df = pd.DataFrame(self.calibration_results)
        
        calculated_transmissivities = []
        for _, res in detailed_df.iterrows():
            well_id = str(res['calibrated_well_id'])
            well_info = self._get_well_info(well_id, 'public')
            if well_info.empty or 'ground_elev_m' not in well_info.columns or well_info['ground_elev_m'].isna().all() or 'dynamic_level_m' not in well_info.columns or well_info['dynamic_level_m'].isna().all():
                calculated_transmissivities.append(np.nan)
                continue
            water_level_m = well_info.iloc[0]['ground_elev_m'] - well_info.iloc[0]['dynamic_level_m']
            
            optimal_conductivities = {p.replace('K_', ''): v for p, v in res.items() if p.startswith('K_')}
            
            effective_T = self._calculate_effective_transmissivity(well_id, 'public', optimal_conductivities, water_level_m)
            calculated_transmissivities.append(effective_T)
        
        detailed_df['effective_T_m2d'] = calculated_transmissivities

        detailed_path = os.path.join(self.results_dir, 'detailed_calibration_results_by_well.csv')
        detailed_df.to_csv(detailed_path, index=False)
        print(f"-> Detailed results (including effective T) saved in: {detailed_path}")

        aggregated_summary = {'hydraulic_parameters': {}}
        k_vals = {}
        s_vals = [v for v in detailed_df['S'] if pd.notna(v)]
        
        for col in detailed_df.columns:
            if col.startswith('K_'):
                formation = col.replace('K_', '')
                k_vals[formation] = [v for v in detailed_df[col] if pd.notna(v)]

        if s_vals:
            aggregated_summary['hydraulic_parameters']['Storage_S'] = {
                'median': np.median(s_vals), 'mean': np.mean(s_vals), 'std_dev': np.std(s_vals),
                'percentile_25': np.percentile(s_vals, 25), 'percentile_75': np.percentile(s_vals, 75), 'n_samples': len(s_vals)
            }

        aggregated_summary['hydraulic_parameters']['Conductivity_K_md'] = {}
        for formation, vals in k_vals.items():
            if vals:
                aggregated_summary['hydraulic_parameters']['Conductivity_K_md'][formation] = {
                    'median': np.median(vals), 'mean': np.mean(vals), 'std_dev': np.std(vals),
                    'percentile_25': np.percentile(vals, 25), 'percentile_75': np.percentile(vals, 75), 'n_samples': len(vals)
                }
        
        summary_path = os.path.join(self.results_dir, 'aggregated_hydraulic_parameters_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(aggregated_summary, f, indent=4)
        print(f"-> Aggregated parameters summary saved in: {summary_path}")

        if self.generate_summary_plots:
            print("\n--- 5. Generating Summary and Analysis Plots ---")

            readme_content = """
# Summary Plot Data

The data required to generate the summary plots (`plot_scatter_Q`, `plot_boxplot_K`, `plot_map_T`, `plot_correlation_heatmap`, `plot_error_map`) are mainly located in two places:

1.  **`detailed_calibration_results_by_well.csv`**:
    -   Location: In the main results directory (one level up from this folder).
    -   Content: Calibration results for each well, including `K`, `S`, `effective_T`, `final_error`, etc.
    -   Used by: All summary plots.

2.  **Input Well Files (Shapefiles)**:
    -   Location: In the `data/` directory of the project.
    -   Files: `public_supply_wells.shp` and `private_supply_wells.shp`.
    -   Content: Geospatial information, reported flow rates (`Q_m3d`), etc.
    -   Used by: `plot_scatter_Q`, `plot_map_T`, `plot_error_map`.

To recreate the plots, load these files into a Python script and use the corresponding visualization functions from the `src.visualization` module.
"""
            readme_path = os.path.join(self.summary_data_dir, 'README.md')
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)

            try:
                plot_scatter_Q(detailed_df, self.public_wells, self.results_dir)
                plot_boxplot_K(detailed_df, self.results_dir)
                plot_map_T(detailed_df, self.public_wells, self.results_dir)
                plot_correlation_heatmap(detailed_df, self.results_dir)
                plot_error_map(detailed_df, self.public_wells, self.results_dir)
                plot_iteration_history(self.iteration_summary_history, self.results_dir)

            except Exception as e:
                print(f"WARNING: Error occurred while generating summary plots: {e}")
        
        print("\n--- Conductivity (K) Summary in m/d (Median) ---")
        for formation, stats in aggregated_summary.get('hydraulic_parameters', {}).get('Conductivity_K_md', {}).items():
            print(f"  - {formation}: {stats['median']:.4f}")
        
        s_stats = aggregated_summary.get('hydraulic_parameters', {}).get('Storage_S')
        if s_stats:
            print(f"\n--- Storage Coefficient (S) Summary (Median) ---")
            print(f"  - Storage: {s_stats['median']:.2e}")

    def execute(self):
        """
        Executes main workflow based on operation mode.
        """
        print(f"--- Starting execution in mode: '{self.operation_mode}' ---")
        
        self.extract_recovery_events()

        if self.operation_mode == 'filtering_only':
            print("\n--- 'filtering_only' mode completed. ---")
            print("Events have been extracted and filtered. Intermediate results saved.")
            return 

        if self.operation_mode == 'complete':
            self.execute_complete_calibration()
            self.save_and_aggregate_results()
        else:
            raise ValueError(f"Operation mode '{self.operation_mode}' not recognized.")

        print("\n--- 'complete' workflow finished. ---")
