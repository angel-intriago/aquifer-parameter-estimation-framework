import numpy as np
from scipy.optimize import differential_evolution
from typing import Dict, List, Tuple, Any, Callable, Optional

def objective_function_de(params: np.ndarray, args: Tuple) -> float:
    """
    Objective function for calibration with Differential Evolution.
    Calculates the root mean square error between observed and calculated slopes.
    """
    (
        target_well_id,
        param_names,
        neighborhood_info,
        well_events,
        calculate_T_func,  # Callback to T calculation function
        consider_interference, # Parameter to control interference
        fixed_params,      # Optional dictionary with fixed parameters
    ) = args

    # Combine optimization parameters with fixed parameters
    param_dict = dict(zip(param_names, params))
    if fixed_params:
        param_dict.update(fixed_params)

    calibrated_q_target = param_dict[f"Q_{target_well_id}"]
    calibrated_conductivities = {
        p.replace("K_", ""): v for p, v in param_dict.items() if p.startswith("K_")
    }

    central_water_level = neighborhood_info[target_well_id]["water_level_m"]
    T_target = calculate_T_func(
        target_well_id, "public", calibrated_conductivities, central_water_level
    )

    total_interference = 0.0
    if consider_interference:
        for neighbor_id, info in neighborhood_info.items():
            if neighbor_id == target_well_id:
                continue

            T_neighbor = calculate_T_func(
                neighbor_id, info["type"], calibrated_conductivities, info["water_level_m"]
            )
            distance = info["distance"]
            # Use calibrated Q of the neighbor if it's being optimized
            q_neighbor = param_dict[f"Q_{neighbor_id}"]

            if T_neighbor > 1e-5 and distance > 1.0:
                if "S" not in param_dict:
                    raise ValueError("'S' parameter not found in optimization or fixed parameters.")
                
                total_interference += (q_neighbor / T_neighbor) * np.exp(
                    -distance ** 2 * param_dict["S"] / (4 * T_neighbor)
                )

    calculated_slopes = (2.303 / (4 * np.pi * T_target)) * (
        calibrated_q_target + T_target * total_interference
    )
    observed_slopes = well_events["slope"].values
    error = np.sqrt(np.mean((observed_slopes - calculated_slopes) ** 2))

    return error

def execute_de_optimization(
    bounds: List[Tuple[float, float]],
    args: Tuple,
    maxiter: int,
    popsize: int,
    seed: int,
    polish: bool,
    updating: str,
    workers: int,
    fixed_params: Optional[Dict[str, float]] = None,
) -> Any:
    """
    Executes optimization with Differential Evolution.

    Args:
        bounds: Limits for each optimization parameter.
        args: Additional arguments for the objective function.
        maxiter: Maximum iterations.
        popsize: Population size.
        seed: Seed for reproducibility.
        polish: Whether to polish the result.
        updating: Population update strategy.
        workers: Number of workers for parallelization.
        fixed_params: Optional dictionary of parameters to keep fixed.

    Returns:
        The OptimizeResult object returned by differential_evolution.
    """
    num_params_to_optimize = len(bounds)
    print(f"  Optimizing {num_params_to_optimize} parameters with Differential Evolution...")
    if fixed_params:
        print(f"  Keeping {len(fixed_params)} parameters fixed: {list(fixed_params.keys())}")

    # Add fixed parameters to the arguments tuple for the objective function
    args_for_objective_function = args + (fixed_params,)

    opt_result = differential_evolution(
        objective_function_de,
        bounds=bounds,
        args=(args_for_objective_function,),
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=polish,
        updating=updating,
        workers=workers,
    )

    return opt_result
