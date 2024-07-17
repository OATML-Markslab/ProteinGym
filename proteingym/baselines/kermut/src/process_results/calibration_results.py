"""Compute and save calibration metrics"""

from src.process_results.utils import (compute_all_calibration_metrics,
                                       compute_calibration_plot_values)


def main():
    model_names = [
        "kermut",
        "kermut_no_m_constant_mean",
    ]
    split_methods = [
        "fold_random_5",
        "fold_modulo_5",
        "fold_contiguous_5",
        "fold_rand_multiples",
        "domain",
    ]

    # Compute and save calibration metrics
    for model_name in model_names:
        for split_method in split_methods:
            # Compute ECE, ENCE, and cv. For metric barplots.
            compute_all_calibration_metrics(
                model_name=model_name, split_method=split_method, limit_n=False
            )
            # Computes RMV, RMSE. For boxplots.
            compute_calibration_plot_values(
                model_name=model_name,
                split_method=split_method,
                calibration_method="error-based",
                limit_n=False,
            )


if __name__ == "__main__":
    main()
