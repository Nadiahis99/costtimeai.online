import numpy as np


MANAGER_EXPERIENCE_BASELINE = 10.0
MANAGER_EXPERIENCE_MIN = 1.0
MANAGER_EXPERIENCE_MAX = 30.0


def align_model_features(df, model):
    if hasattr(model, "feature_names_in_"):
        return df.reindex(columns=model.feature_names_in_, fill_value=0)
    return df


def adjust_duration_ratio_for_manager_experience(raw_duration_ratio, manager_experience):
    """Apply a monotonic domain correction for manager experience."""
    duration_ratio = float(raw_duration_ratio)
    experience = float(
        np.clip(manager_experience, MANAGER_EXPERIENCE_MIN, MANAGER_EXPERIENCE_MAX)
    )
    schedule_gap = duration_ratio - 1.0

    if experience <= MANAGER_EXPERIENCE_BASELINE:
        missing_years = MANAGER_EXPERIENCE_BASELINE - experience
        low_experience_factor = 1.0 + (0.06 * missing_years)

        if schedule_gap >= 0:
            return 1.0 + (schedule_gap * low_experience_factor)

        early_finish_factor = max(0.25, 1.0 - (0.06 * missing_years))
        return 1.0 + (schedule_gap * early_finish_factor)

    extra_years = experience - MANAGER_EXPERIENCE_BASELINE

    if schedule_gap >= 0:
        high_experience_factor = 1.0 / (1.0 + (0.12 * extra_years))
        return 1.0 + (schedule_gap * high_experience_factor)

    early_finish_factor = min(1.75, 1.0 + (0.04 * extra_years))
    return 1.0 + (schedule_gap * early_finish_factor)
