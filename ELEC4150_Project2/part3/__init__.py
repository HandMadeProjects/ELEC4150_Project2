"""
ELEC4150/8150 Project 2 — Part 3: Microphone Array & Source Localisation
========================================================================
Modules:
  - microphone_model: Circular array geometry, distance calculations,
                      inverse-square power law attenuation, array simulation.
  - delay: Sub-sample propagation delay via Fourier phase shift,
           speed of sound propagation delay calculations.
  - localisation: GCC-PHAT TDOA estimation, Non-linear Least Squares (NLS)
                  source multilateration, SRP-PHAT acoustic spatial map,
                  accuracy evaluation.
"""

from part3.microphone_model import (
    create_circular_array,
    generate_random_source,
    calculate_distances,
    apply_inverse_square_attenuation,
    simulate_microphone_array,
)
from part3.delay import (
    fractional_delay,
    calculate_propagation_delays,
    apply_propagation_delays,
)
from part3.localisation import (
    estimate_tdoa_gcc_phat,
    compute_pairwise_tdoas,
    localise_source_nls,
    steered_response_power_map,
    evaluate_localisation_accuracy,
)

__all__ = [
    "create_circular_array",
    "generate_random_source",
    "calculate_distances",
    "apply_inverse_square_attenuation",
    "simulate_microphone_array",
    "fractional_delay",
    "calculate_propagation_delays",
    "apply_propagation_delays",
    "estimate_tdoa_gcc_phat",
    "compute_pairwise_tdoas",
    "localise_source_nls",
    "steered_response_power_map",
    "evaluate_localisation_accuracy",
]
