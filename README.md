# Metaheuristic-driven Aquifer Calibration Framework

Automated framework using Differential Evolution metaheuristics to estimate aquifer hydraulic parameters (Hydraulic Conductivity $K$, Storage Coefficient $S$) and pumping rates ($Q$) in high-interference urban settings.

The method analyzes water level recovery curves and accounts for variable saturated thickness and interference from neighboring wells.

## Purpose

1. **Extract recovery events**: Automatically identify and filter valid water level recovery periods from high-resolution pressure transducer data.
2. **Model interference**: Account for drawdown effects of multiple pumping wells in the neighborhood.
3. **Optimize parameters**: Use Differential Evolution to find the best-fit hydraulic parameters and actual pumping rates that explain the observed recovery slopes.
4. **Ensure spatial consistency**: Iteratively refine parameters across interconnected neighborhoods.

## Repository structure

```
├── main.py                          # Entry point for the full pipeline
├── example_run.py                   # Quick test using demo_data/
├── requirements.txt                 # Pinned dependencies
├── .gitignore
├── LICENSE                          # MIT License
├── README.md
├── src/
│   ├── __init__.py
│   ├── advanced_hydro_calibrator.py # Core calibration logic
│   ├── data_loading.py              # CSV and shapefile loading
│   ├── visualization.py             # Result maps and diagnostic plots
│   ├── calibration/                 # Optimization logic
│   ├── event_extraction/            # Recovery event identification and filtering
│   └── utils/                       # Helper functions
└── demo_data/
    ├── head_time_series.csv                 # Water level time series (1 well)
    ├── public_supply_wells.shp              # Public pumping wells (3 wells)
    ├── private_supply_wells.shp             # Private pumping wells (10 wells)
    ├── stratigraphy_public_wells.csv        # Stratigraphy (public wells)
    └── stratigraphy_private_wells.csv       # Stratigraphy (private wells)

> **Note:** Each shapefile includes companion files (`.dbf`, `.shx`, `.prj`, etc.).

## Requirements

Python 3.8+ with:

- `pandas`
- `numpy`
- `geopandas` (requires `fiona` and `shapely`)
- `scipy`
- `matplotlib`
- `seaborn`

Install:

```bash
pip install pandas numpy geopandas scipy matplotlib seaborn
```

## Input data format

### 1. Water level time series — `head_time_series.csv`

| Column       | Type     | Description                         |
|--------------|----------|-------------------------------------|
| `well_id`    | str      | Unique well identifier              |
| `datetime`   | datetime | Measurement timestamp               |
| `head_m`     | float    | Piezometric head (m)                |

### 2. Public wells — `public_supply_wells.shp`

| Column      | Type   | Description                    |
|-------------|--------|--------------------------------|
| `well_id`   | str    | Unique well identifier         |
| `ground_elev_m` | float | Ground elevation (m.a.s.l.)        |
| `static_level_m` | float | Static water level (m)     |
| `dynamic_level_m` | float | Dynamic water level (m)   |
| `pump_rate_m3h` | float | Average pumping rate (m³/h)  |
| `geometry`  | point  | Location (UTM 23S, EPSG:31983)|

### 3. Private wells — `private_supply_wells.shp`

| Column      | Type   | Description                    |
|-------------|--------|--------------------------------|
| `well_id`   | str    | Unique well identifier         |
| `pump_rate_m3h` | float | Pumping rate (m³/h)          |
| `ground_elev_m` | float | Ground elevation (m)        |
| `geometry`  | point  | Location (UTM 23S, EPSG:31983)|

### 4. Stratigraphy — `stratigraphy_public_wells.csv` / `stratigraphy_private_wells.csv`

Lithological information per well (depth, geological formation, thickness, etc.). All columns are used by the pipeline.



## Usage

1. **Prepare data**: For a quick test, `demo_data/` is ready to use with `example_run.py`. For the full pipeline, place the complete dataset in `data/`.
2. **Configure `main.py`**: Verify paths point to the correct files. Adjust parameters such as:
   - `consider_interference` (`True`/`False`): whether to model interference between neighboring wells.
   - Shapefile and CSV paths.
   - Filtering parameters (`min_points`, `max_slope`, etc.) to adjust event detection sensitivity.
3. **Run**:
   ```bash
   python example_run.py   # quick demo
   python main.py          # full pipeline (requires full dataset)
   ```

The script creates a timestamped subfolder in `results/` with all generated CSVs and plots.

## Outputs

After a complete run, `results/YYYY-MM-DD_HH-MM-SS/` contains:

| File | Description |
|------|-------------|
| `extracted_recovery_events.csv` | Detected and filtered recovery events |
| `detailed_calibration_results_by_well.csv` | Per-well results (K, S, Q, T, error) |
| `aggregated_hydraulic_parameters_summary.json` | Aggregated hydraulic parameter summary |
| `filtering_summary_by_stage.json` | Per-stage filtering statistics |
| `discards_stage_*.csv` | Events discarded at each filter stage |
| `3_result_boxplot_K.png` | Hydraulic conductivity boxplot by formation |
| `4_result_map_T.png` | Transmissivity map |
| `5_result_scatter_Q.png` | Pumping rate scatter plot |
| `6_analysis_parameter_correlation.png` | Parameter correlation matrix |
| `7_analysis_error_map.png` | Calibration error map |
| `8_parameter_evolution_iteration.png` | Parameter evolution across iterations |
| `neighborhood_figures/` | Interference plots per neighborhood |
| `recovery_event_figures/` | Recovery curves for each event |

## Quick test

```bash
python example_run.py
```

Runs a simplified version of the pipeline with the demo dataset (single central well with 3 public neighbours).

## Dataset notes

The repository includes a **demo dataset** (`demo_data/`) containing only the
neighbourhood of a single target well (within the 2000 m search radius used by
the calibration code): 3 public wells and 10 private wells. Coordinates were
randomised preserving the radial distance to the target well while assigning
arbitrary angles, so the interference calculation produces identical results
but the spatial pattern cannot be matched to the published paper's map.

The **full dataset** is not included in the repository. If you have access to the
complete dataset, place it in a `data/` directory and use `main.py` instead.

## Troubleshooting

| Problem | Likely cause | Solution |
|---------|-------------|----------|
| `KeyError: 'well_id'` on load | Missing expected column | Verify column names (see *Input data format*) |
| Error reading shapefile | Missing `.dbf` or `.shx` file | Ensure all shapefile companion files are present |
| `ModuleNotFoundError: fiona` | Missing geopandas dependency | `pip install fiona` or reinstall geopandas |
| Wrong encoding in columns | Shapefile encoding differs from UTF-8 | Check `.cpg` or set `encoding=` in `data_loading.py` |
| Calibration does not converge | Too few recovery events for a well | Adjust filtering parameters in `main.py` |

## License

MIT. See [LICENSE](LICENSE).

## Authorship

- **Angel Intriago**: Software Development, Data Curation, Validation, Writing.
- **Bruno Conicelli**: Conceptualization, Methodology, Supervision.

## Code availability

- **Name**: Metaheuristic-driven Aquifer Calibration Framework
- **Developers**: Angel Intriago, Bruno Conicelli, 
- **Year**: 2026
- **Software required**: Python 3.8+
- **Language**: Python
- **URL**: https://github.com/angel-intriago/aquifer-parameter-estimation-framework
- **License**: MIT
