import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

def plot_recovery_event(
    group_df,
    event_df,
    slope,
    well_id,
    event_date,
    output_dir
):
    """
    Generates a plot of a single recovery event.
    """
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot all group data (day or night)
    ax.plot(group_df['datetime'], group_df['head_m'], color='lightgrey', label="Observed Water Level")

    # Highlight recovery period
    ax.plot(event_df['datetime'], event_df['head_m'], 'bo-', label='Identified Recovery Period')

    # Plot logarithmic fit
    t_log = np.log10(event_df['t_min'][event_df['t_min'] > 0])
    # Reconstruct line to plot: y = m*x + c, where c = y_mean - m*x_mean
    adjusted_level = slope * t_log + (event_df['head_m'][event_df['t_min'] > 0].mean() - slope * t_log.mean())
    
    ax.plot(
        event_df['datetime'][event_df['t_min'] > 0],
        adjusted_level,
        'r--',
        linewidth=2,
        label=f'Logarithmic Fit (Slope A = {slope:.3f})'
    )

    ax.set_xlabel('Time of Day')
    ax.set_ylabel("Water Level (elevation, m)")
    ax.set_title(f"Methodology: Recovery Slope Extraction\nWell {well_id} - {event_date.strftime('%Y-%m-%d')}")
    ax.grid(True, linestyle='--')
    ax.legend()
    
    # Date format on x axis
    from matplotlib.dates import DateFormatter
    ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))

    filename = f"event_{well_id}_{event_date.strftime('%Y-%m-%d')}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_scatter_Q(detailed_df, public_wells, results_dir):
    """
    Generates a scatter plot comparing reported vs. calibrated flow rate.
    """
    reported_q_list = []
    calibrated_q_list = []

    for _, row in detailed_df.iterrows():
        well_id = str(row['calibrated_well_id'])
        
        # Get calibrated Q
        calibrated_q = row.get(f'Q_{well_id}')
        if calibrated_q is None:
            continue
            
        # Get reported Q
        reported_q_info = public_wells[public_wells['well_id'] == well_id]
        if reported_q_info.empty:
            continue
        reported_q = reported_q_info.iloc[0]['Q_m3d']
        
        if pd.notna(reported_q) and pd.notna(calibrated_q) and reported_q > 0 and calibrated_q > 0:
            reported_q_list.append(reported_q)
            calibrated_q_list.append(calibrated_q)

    if not reported_q_list:
        print("WARNING: No flow rate data to generate scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(reported_q_list, calibrated_q_list, alpha=0.7, edgecolors='k')
    
    ax.set_xlabel('Reported Flow Rate, Q (m³/d) - Log Scale')
    ax.set_ylabel('Calibrated Flow Rate, Q (m³/d) - Log Scale')
    ax.set_title('Result: Reported vs. Calibrated Flow Rate')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    lim_min = min(min(reported_q_list), min(calibrated_q_list)) * 0.8
    lim_max = max(max(reported_q_list), max(calibrated_q_list)) * 1.2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', label='1:1 Line')
    
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.grid(True, which="both", ls="--")
    ax.legend()
    
    output_path = os.path.join(results_dir, '5_result_scatter_Q.png')
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"-> Flow rate scatter plot saved in: {output_path}")

def plot_boxplot_K(detailed_df, results_dir):
    """
    Generates a boxplot of Hydraulic Conductivity (K) values by formation.
    """
    k_cols = [col for col in detailed_df.columns if col.startswith('K_')]
    if not k_cols:
        print("WARNING: No conductivity data to generate boxplot.")
        return
        
    df_k = detailed_df[k_cols].copy()
    df_k.rename(columns=lambda c: c.replace('K_', ''), inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_k.plot(kind='box', ax=ax)
    
    ax.set_ylabel('Hydraulic Conductivity, K (m/d) - Log Scale')
    ax.set_title('Result: Calibrated Hydraulic Conductivity (K) Distribution by Formation')
    ax.set_yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = os.path.join(results_dir, '3_result_boxplot_K.png')
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"-> Conductivity boxplot saved in: {output_path}")

def plot_map_T(detailed_df, public_wells, results_dir):
    """
    Generates a map of calibrated Transmissivity (T).
    """
    if 'effective_T_m2d' not in detailed_df.columns:
        print("WARNING: No transmissivity data to generate map.")
        return

    calibrated_wells_ids = detailed_df['calibrated_well_id'].astype(str).tolist()
    
    # GeoDataFrame of calibrated wells
    gdf_calibrated = public_wells[public_wells['well_id'].isin(calibrated_wells_ids)].copy()
    gdf_calibrated = gdf_calibrated.merge(detailed_df, left_on='well_id', right_on='calibrated_well_id')

    # GeoDataFrame of non-calibrated wells
    gdf_non_calibrated = public_wells[~public_wells['well_id'].isin(calibrated_wells_ids)].copy()

    if gdf_calibrated.empty:
        print("WARNING: No calibrated wells with geometry found to generate T map.")
        return

    # --- Load and transform Municipality Shapefile ---
    municipality_gdf = None
    for candidate in ['data/study_area.shp', 'demo_data/study_area.shp']:
        if os.path.exists(candidate):
            try:
                municipality_gdf = gpd.read_file(candidate).to_crs(gdf_calibrated.crs)
            except Exception:
                pass
            break

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot non-calibrated wells
    if not gdf_non_calibrated.empty:
        gdf_non_calibrated.plot(ax=ax, marker='o', color='grey', markersize=30, label='Non-calibrated Wells', zorder=2)

    # Plot calibrated wells
    scatter = ax.scatter(
        x=gdf_calibrated.geometry.x,
        y=gdf_calibrated.geometry.y,
        c=gdf_calibrated['effective_T_m2d'],
        cmap='viridis',
        s=100, # markersize
        edgecolor='black',
        norm=LogNorm(), # Apply log scale to colors
        zorder=3
    )

    # Plot municipality background
    if municipality_gdf is not None:
        municipality_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=4)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.07, shrink=0.8)
    cbar.set_label("Transmissivity, T (m²/d)")
    
    # Annotate labels
    for x, y, label in zip(gdf_calibrated.geometry.x, gdf_calibrated.geometry.y, gdf_calibrated['well_id']):
        ax.text(x, y, label, fontsize=8, ha='right', zorder=5)

    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title('Result: Spatial Distribution of Calibrated Transmissivity (T)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_path = os.path.join(results_dir, '4_result_map_T.png')
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"-> Transmissivity map saved in: {output_path}")

def plot_correlation_heatmap(detailed_df, results_dir):
    """
    Generates a correlation matrix of calibrated parameters.
    """
    param_cols = [col for col in detailed_df.columns if col.startswith('K_') or col.startswith('Q_') or col == 'S' or col == 'effective_T_m2d']
    if len(param_cols) < 2:
        print("WARNING: Not enough parameters to generate a correlation matrix.")
        return

    df_params = detailed_df[param_cols].copy()
    
    # Rename columns for clarity in the plot
    df_params.rename(columns=lambda c: c.replace('K_', 'K ').replace('effective_T_m2d', 'Effective T'), inplace=True)
    # Remove well ID from Q column for a more generic name
    df_params.rename(columns=lambda c: 'Calibrated Q' if c.startswith('Q_') else c, inplace=True)


    corr_matrix = df_params.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=.5)
    
    ax.set_title('Analysis: Correlation Matrix of Calibrated Parameters')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(results_dir, '6_analysis_parameter_correlation.png')
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"-> Parameter correlation matrix saved in: {output_path}")

def plot_error_map(detailed_df, public_wells, results_dir):
    """
    Generates a map of final calibration errors by well.
    """
    if 'final_error' not in detailed_df.columns:
        print("WARNING: No error data to generate map.")
        return

    calibrated_wells_ids = detailed_df['calibrated_well_id'].astype(str).tolist()
    
    gdf_calibrated = public_wells[public_wells['well_id'].isin(calibrated_wells_ids)].copy()
    gdf_calibrated = gdf_calibrated.merge(detailed_df, left_on='well_id', right_on='calibrated_well_id')

    if gdf_calibrated.empty:
        print("WARNING: No calibrated wells with geometry found to generate error map.")
        return

    municipality_gdf = None
    for candidate in ['data/study_area.shp', 'demo_data/study_area.shp']:
        if os.path.exists(candidate):
            try:
                municipality_gdf = gpd.read_file(candidate).to_crs(gdf_calibrated.crs)
            except Exception:
                pass
            break

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Use a minimum value for zero errors if using log scale
    min_error_val = detailed_df['final_error'][detailed_df['final_error'] > 0].min() * 0.1 if (detailed_df['final_error'] > 0).any() else 1e-9
    gdf_calibrated['error_plot'] = gdf_calibrated['final_error'].clip(lower=min_error_val)

    scatter = ax.scatter(
        x=gdf_calibrated.geometry.x,
        y=gdf_calibrated.geometry.y,
        c=gdf_calibrated['error_plot'],
        cmap='plasma',
        s=100,
        edgecolor='black',
        norm=LogNorm(),
        zorder=3
    )

    if municipality_gdf is not None:
        municipality_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=4)
    
    cbar = plt.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.07, shrink=0.8)
    cbar.set_label("Final Calibration Error (RMSE)")
    
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title('Analysis: Spatial Distribution of Calibration Error')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_path = os.path.join(results_dir, '7_analysis_error_map.png')
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"-> Calibration error map saved in: {output_path}")

def plot_neighborhood_drawdown(
    target_coords,
    neighbors_gdf,
    interference_percentages,
    search_radius_m,
    output_dir,
    well_id
):
    """
    Generates a map of the well neighborhood showing drawdown contribution percentages.
    All coordinates are relative to the target well.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Extract self contribution and neighbor contributions
    self_contribution = interference_percentages.pop(well_id, 0.0)
    neighbor_percentages = interference_percentages

    # Target well at the center
    ax.plot(0, 0, 'k*', markersize=20, label=f'Target Well: {well_id}', zorder=10)

    # Search radius
    circle = plt.Circle((0, 0), radius=search_radius_m, color='grey', fill=False, linestyle='--', label=f'Search Radius ({search_radius_m} m)')
    ax.add_patch(circle)

    percentages = []
    if not neighbors_gdf.empty and neighbor_percentages:
        valid_ids = [nid for nid in neighbor_percentages.keys() if nid in neighbors_gdf['unified_id'].values]
        percentages = np.array([neighbor_percentages[nid] for nid in valid_ids])
        
        if len(percentages) > 0:
            cmap = plt.get_cmap('plasma')
            # Normalize from 0 to max percentage found in neighbors
            norm = plt.Normalize(vmin=0, vmax=percentages.max() if percentages.max() > 0 else 1)

            # Plot neighbors
            for neighbor_id in valid_ids:
                neighbor_series = neighbors_gdf.loc[neighbors_gdf['unified_id'] == neighbor_id].iloc[0]
                neighbor_geom = neighbor_series.geometry
                rel_x = neighbor_geom.x - target_coords.x
                rel_y = neighbor_geom.y - target_coords.y
                
                percentage = neighbor_percentages[neighbor_id]
                color = cmap(norm(percentage))
                
                ax.plot(rel_x, rel_y, 'o', markersize=12, color=color)
                ax.text(rel_x + 25, rel_y + 25, f'{neighbor_id}\n({percentage:.1f}%)', fontsize=9, ha='left')

    # Create a colorbar
    if len(percentages) > 0:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Interference Contribution (%)')

    # Create a clean legend
    legend_elements = [
        Line2D([0], [0], marker='*', color='k', label=f'Target Well ({self_contribution:.1f}%)', markersize=15, linestyle='None'),
        Line2D([0], [0], marker='o', mfc='grey', mec='grey', label='Neighbor Well', markersize=12, linestyle='None'),
        Line2D([0], [0], color='grey', lw=2, linestyle='--', label=f'Search Radius ({search_radius_m} m)')
    ]
    ax.legend(handles=legend_elements, loc='best')

    ax.set_xlabel('Relative Easting (m)')
    ax.set_ylabel('Relative Northing (m)')
    ax.set_title(f'Neighborhood Interference Analysis for Well {well_id}')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':')
    
    plt.tight_layout()
    
    neighborhoods_dir = os.path.join(output_dir, 'neighborhood_figures')
    os.makedirs(neighborhoods_dir, exist_ok=True)
    
    filename = f"neighborhood_{well_id}.png"
    output_path = os.path.join(neighborhoods_dir, filename)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"-> Neighborhood interference analysis plot saved in: {output_path}")

def plot_comparative_flow(well_id, reported_q, calibrated_q, output_dir, custom_filename=None):
    """
    Generates a bar chart comparing reported and calibrated flow rate for a single well.
    """
    if reported_q is None or calibrated_q is None or reported_q == 0 or np.isnan(reported_q) or np.isnan(calibrated_q):
        return

    os.makedirs(output_dir, exist_ok=True)

    # Data for plot
    labels = ['Reported', 'Calibrated']
    values = [reported_q, calibrated_q]
    
    # Calculate percentage change
    pct_change = ((calibrated_q - reported_q) / reported_q) * 100
    
    # Bar colors
    reported_color = 'grey'
    calibrated_color = 'green' if calibrated_q >= reported_q else 'red'

    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(labels, values, color=[reported_color, calibrated_color])
    
    # Add labels with values
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')

    # Titles and labels
    ax.set_ylabel('Flow Rate (m³/d)')
    ax.set_title(f'Flow Rate Comparison for Well {well_id}')
    
    # Annotation of percentage change
    sign = '+' if pct_change >= 0 else ''
    ax.text(0.5, 0.9, f'Change: {sign}{pct_change:.1f}% ', 
            horizontalalignment='center', 
            verticalalignment='center', 
            transform=ax.transAxes,
            fontsize=14,
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    if custom_filename:
        filename = custom_filename
    else:
        filename = f"comparative_q_{well_id}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_iteration_history(iteration_summary_history, output_dir):
    """
    Generates plots of parameter evolution (K, S) and error across calibration iterations.
    """
    if not iteration_summary_history:
        print("WARNING: No iteration summary history to plot.")
        return

    history_df = pd.DataFrame(iteration_summary_history)
    iter_nums = history_df['iter_num']

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # Plot 1: Error Evolution
    axes[0].plot(iter_nums, history_df['average_error'], marker='o', linestyle='-', color='red')
    axes[0].set_ylabel('Average Error (RMSE)')
    axes[0].set_title('Average Error Evolution by Iteration')
    axes[0].grid(True, linestyle='--')

    # Plot 2: S Evolution
    if 'S_median' in history_df.columns:
        axes[1].plot(iter_nums, history_df['S_median'], marker='o', linestyle='-', color='blue')
        axes[1].set_ylabel('S Median')
        axes[1].set_title('S Median Evolution by Iteration')
        axes[1].set_yscale('log')
        axes[1].grid(True, linestyle='--')

    # Plot 3: K Evolution by Formation
    k_cols = [col for col in history_df.columns if col.startswith('K_') and col.endswith('_median')]
    if k_cols:
        for col in k_cols:
            formation = col.replace('K_', '').replace('_median', '')
            axes[2].plot(iter_nums, history_df[col], marker='o', linestyle='-', label=f'K {formation}')
        axes[2].set_ylabel('K Median (m/d)')
        axes[2].set_title('K Median Evolution by Formation by Iteration')
        axes[2].set_yscale('log')
        axes[2].grid(True, linestyle='--')
        axes[2].legend()

    axes[2].set_xlabel('Iteration Number')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '8_parameter_evolution_iteration.png')
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"-> Parameter evolution plot saved in: {output_path}")
