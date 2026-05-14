import pandas as pd
import geopandas as gpd
from typing import Dict, Union

# Shapefile column name mapping (truncated to 10 chars by ESRI spec)
_SHAPEFILE_COL_MAP = {
    'ground_ele': 'ground_elev_m',
    'static_lev': 'static_level_m',
    'dynamic_le': 'dynamic_level_m',
    'pump_rate_': 'pump_rate_m3h',
}

def _fix_shapefile_cols(gdf):
    return gdf.rename(columns=_SHAPEFILE_COL_MAP)

def load_hydrogeological_data(
    observed_df_path: str,
    private_wells_path: str,
    public_wells_path: str,
    private_stratigraphy_path: str,
    public_stratigraphy_path: str
) -> Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]]:
    """
    Loads all data necessary for hydrogeological calibration.

    Args:
        observed_df_path: Path to CSV with water level observation data.
        private_wells_path: Path to Shapefile of private wells.
        public_wells_path: Path to Shapefile of public wells.
        private_stratigraphy_path: Path to CSV with stratigraphy data for private wells.
        public_stratigraphy_path: Path to CSV with stratigraphy data for public wells.

    Returns:
        A dictionary with loaded and pre-processed DataFrames and GeoDataFrames.
    """
    print("--- 1. Loading Data ---")

    # Load and process observation data
    df_obs = pd.read_csv(observed_df_path)
    df_obs['well_id'] = df_obs['well_id'].astype(str)
    df_obs['datetime'] = pd.to_datetime(df_obs['datetime'])
    df_obs = df_obs.groupby(['well_id', 'datetime']).mean().reset_index()
    observed_df = df_obs.sort_values(by=['well_id', 'datetime']).reset_index(drop=True)

    # Load and process private wells
    private_wells = gpd.read_file(private_wells_path)
    private_wells = _fix_shapefile_cols(private_wells)
    private_wells['Q_m3d'] = private_wells.get('pump_rate_m3h', 0) * 24
    private_wells['well_id'] = private_wells['well_id'].astype(str)

    # Load and process public wells
    public_wells = gpd.read_file(public_wells_path)
    public_wells = _fix_shapefile_cols(public_wells)
    public_wells['Q_m3d'] = public_wells.get('pump_rate_m3h', 0).fillna(0)
    public_wells['well_id'] = public_wells['well_id'].astype(str)

    # Load stratigraphy data
    private_stratigraphy = pd.read_csv(private_stratigraphy_path, encoding='latin-1')
    public_stratigraphy = pd.read_csv(public_stratigraphy_path, encoding='latin-1')

    print("-> Data loaded successfully.")

    return {
        "observed_df": observed_df,
        "private_wells": private_wells,
        "public_wells": public_wells,
        "private_stratigraphy": private_stratigraphy,
        "public_stratigraphy": public_stratigraphy,
    }
