"""예측 모델을 사용해 운행 조건 후보를 비교하고 추천한다."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ev_energy.data import BASIC_FEATURES, EXPANDED_FEATURES
else:
    from .data import BASIC_FEATURES, EXPANDED_FEATURES

MODE_LIMITS = {"fast": 1.00, "balanced": 1.10, "saver": 1.20}
CONTROL_FEATURES = [
    "speed_kmh",
    "hvac_power_kw",
    "driving_style_index",
    "tire_pressure_bar",
]


def total_energy_kwh(consumption_kwh_per_100km: float, distance_km: float) -> float:
    """소비율과 거리에서 총소비량을 계산한다."""
    return consumption_kwh_per_100km * distance_km / 100


def energy_uncertainty_kwh(model_mae: float, distance_km: float) -> float:
    """MAE를 같은 거리의 총소비량 오차로 환산한다."""
    return model_mae * distance_km / 100


def _values_between(lower: float, upper: float, step: float) -> np.ndarray:
    start = np.ceil(lower / step) * step
    end = np.floor(upper / step) * step
    if start > end:
        return np.array([])
    return np.round(np.arange(start, end + step / 2, step), 6)


def _candidate_values(
    train_frame: pd.DataFrame,
    manufacturer_tire_range: tuple[float, float],
) -> dict[str, np.ndarray]:
    values = {
        "speed_kmh": _values_between(
            train_frame["speed_kmh"].min(), train_frame["speed_kmh"].max(), 5.0
        ),
        "hvac_power_kw": _values_between(
            train_frame["hvac_power_kw"].min(), train_frame["hvac_power_kw"].max(), 0.5
        ),
        "driving_style_index": _values_between(
            train_frame["driving_style_index"].min(),
            train_frame["driving_style_index"].max(),
            0.05,
        ),
    }
    lower = max(train_frame["tire_pressure_bar"].min(), manufacturer_tire_range[0])
    upper = min(train_frame["tire_pressure_bar"].max(), manufacturer_tire_range[1])
    values["tire_pressure_bar"] = _values_between(lower, upper, 0.05)
    if any(len(candidates) == 0 for candidates in values.values()):
        raise ValueError("학습 범위와 제조사 제한에서 만들 수 있는 추천 후보가 없습니다.")
    return values


def _prediction_frame(
    controls: pd.DataFrame,
    train_frame: pd.DataFrame,
    trip_distance_km: float,
    ambient_temp_C: float,
    features: list[str],
) -> pd.DataFrame:
    frame = controls.copy()
    frame["trip_distance_km"] = trip_distance_km
    frame["ambient_temp_C"] = ambient_temp_C
    for feature in EXPANDED_FEATURES:
        if feature not in frame and feature in features:
            frame[feature] = float(train_frame[feature].median())
    return frame[features]


def recommend_trip(
    model: Any,
    train_frame: pd.DataFrame,
    trip_distance_km: float,
    ambient_temp_C: float,
    base_trip_minutes: float,
    profile: dict[str, float],
    mode: str,
    manufacturer_tire_range: tuple[float, float],
    model_mae: float,
    features: list[str] | None = None,
) -> dict[str, float | str | bool]:
    """시간 제약을 지키는 후보 중 총소비량이 가장 낮은 계획을 선택한다."""
    if mode not in MODE_LIMITS:
        raise ValueError(f"알 수 없는 추천 모드입니다: {mode}")
    if trip_distance_km <= 0 or base_trip_minutes <= 0:
        raise ValueError("거리와 기준 시간은 0보다 커야 합니다.")

    features = BASIC_FEATURES if features is None else features
    base_speed = profile.get("speed_kmh", trip_distance_km / base_trip_minutes * 60)
    if base_speed <= 0:
        raise ValueError("기준 평균 속도는 0보다 커야 합니다.")

    values = _candidate_values(train_frame, manufacturer_tire_range)
    controls = pd.DataFrame(
        product(*(values[feature] for feature in CONTROL_FEATURES)),
        columns=CONTROL_FEATURES,
    )
    controls["estimated_trip_minutes"] = (
        base_trip_minutes * base_speed / controls["speed_kmh"]
    )
    controls = controls.loc[
        controls["estimated_trip_minutes"] <= base_trip_minutes * MODE_LIMITS[mode]
    ].copy()
    if controls.empty:
        raise ValueError("선택한 모드의 시간 제한을 만족하는 후보가 없습니다.")

    candidate_inputs = _prediction_frame(
        controls, train_frame, trip_distance_km, ambient_temp_C, features
    )
    controls["predicted_consumption_kwhper100km"] = model.predict(candidate_inputs)
    controls["total_energy_kwh"] = (
        controls["predicted_consumption_kwhper100km"] * trip_distance_km / 100
    )
    best = controls.loc[controls["total_energy_kwh"].idxmin()]

    baseline_controls = pd.DataFrame(
        [
            {
                "speed_kmh": base_speed,
                "hvac_power_kw": profile.get("hvac_power_kw", train_frame["hvac_power_kw"].median()),
                "driving_style_index": profile.get(
                    "driving_style_index", train_frame["driving_style_index"].median()
                ),
                "tire_pressure_bar": profile.get(
                    "tire_pressure_bar", train_frame["tire_pressure_bar"].median()
                ),
            }
        ]
    )
    baseline_rate = float(
        model.predict(
            _prediction_frame(
                baseline_controls, train_frame, trip_distance_km, ambient_temp_C, features
            )
        )[0]
    )
    baseline_energy = total_energy_kwh(baseline_rate, trip_distance_km)
    saving = baseline_energy - float(best["total_energy_kwh"])
    uncertainty = energy_uncertainty_kwh(model_mae, trip_distance_km)

    return {
        "mode": mode,
        "speed_kmh": float(best["speed_kmh"]),
        "hvac_power_kw": float(best["hvac_power_kw"]),
        "driving_style_index": float(best["driving_style_index"]),
        "tire_pressure_bar": float(best["tire_pressure_bar"]),
        "estimated_trip_minutes": float(best["estimated_trip_minutes"]),
        "additional_minutes": float(best["estimated_trip_minutes"] - base_trip_minutes),
        "current_consumption_kwhper100km": baseline_rate,
        "recommended_consumption_kwhper100km": float(best["predicted_consumption_kwhper100km"]),
        "current_total_energy_kwh": baseline_energy,
        "recommended_total_energy_kwh": float(best["total_energy_kwh"]),
        "saving_kwh": saving,
        "saving_percent": saving / baseline_energy * 100 if baseline_energy else 0.0,
        "energy_uncertainty_kwh": uncertainty,
        "saving_is_confident": saving > uncertainty,
    }
