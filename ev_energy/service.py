"""경로·날씨 데이터와 EV 추천 모델을 연결하는 서비스 계층."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import html
import os
from pathlib import Path
from typing import Any

import folium
import httpx
import pandas as pd

from .data import BASIC_FEATURES, load_ev_data, split_ev_data
from .modeling import evaluate_regressor, train_regressor
from .recommendation import MODE_LIMITS, recommend_trip

ORS_BASE_URL = "https://api.openrouteservice.org"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "datas_ml" / "ev_energy_consumption.csv"
DEFAULT_TIRE_RANGE = (2.3, 2.7)


class PredictionError(ValueError):
    """사용자에게 안전하게 표시할 수 있는 예측 오류."""


@dataclass(frozen=True)
class Location:
    label: str
    longitude: float
    latitude: float


@dataclass(frozen=True)
class RoutePrediction:
    start: Location
    destination: Location
    distance_km: float
    duration_minutes: float
    temperature_c: float
    weather_notice: str | None
    map_html: str
    recommendation: dict[str, float | str | bool]


def _api_key() -> str:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists() and not os.getenv("ORS_API_KEY"):
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ORS_API_KEY="):
                os.environ["ORS_API_KEY"] = line.partition("=")[2].strip().strip('"')
                break
    key = os.getenv("ORS_API_KEY", "").strip()
    if not key or key == "your_openrouteservice_api_key":
        raise PredictionError(
            "OpenRouteService API 키가 필요합니다. .env 파일에 ORS_API_KEY를 설정해 주세요."
        )
    return key


def _request_json(client: httpx.Client, url: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = client.get(url, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as error:
        if error.response.status_code in {401, 403}:
            raise PredictionError("OpenRouteService API 키를 확인해 주세요.") from error
        raise PredictionError("외부 경로 서비스를 현재 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.") from error
    except (httpx.HTTPError, ValueError) as error:
        raise PredictionError("외부 경로 서비스를 현재 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.") from error


def _geocode(client: httpx.Client, query: str, key: str, role: str) -> Location:
    data = _request_json(
        client,
        f"{ORS_BASE_URL}/geocode/search",
        params={"text": query, "size": 1},
        headers={"Authorization": key},
    )
    features = data.get("features", [])
    if not features:
        raise PredictionError(f"{role} '{query}'을(를) 찾지 못했습니다. 더 구체적인 주소를 입력해 주세요.")
    feature = features[0]
    coordinates = feature.get("geometry", {}).get("coordinates", [])
    if len(coordinates) < 2:
        raise PredictionError(f"{role} 좌표를 가져오지 못했습니다.")
    properties = feature.get("properties", {})
    return Location(
        label=str(properties.get("label") or query),
        longitude=float(coordinates[0]),
        latitude=float(coordinates[1]),
    )


def _route(client: httpx.Client, start: Location, destination: Location, key: str) -> tuple[float, float, list[list[float]]]:
    data = _request_json(
        client,
        f"{ORS_BASE_URL}/v2/directions/driving-car",
        params={
            "start": f"{start.longitude},{start.latitude}",
            "end": f"{destination.longitude},{destination.latitude}",
        },
        headers={"Authorization": key},
    )
    features = data.get("features", [])
    if not features:
        raise PredictionError("두 지점 사이의 운전 경로를 찾지 못했습니다.")
    feature = features[0]
    summary = feature.get("properties", {}).get("summary", {})
    coordinates = feature.get("geometry", {}).get("coordinates", [])
    distance_m = float(summary.get("distance", 0))
    duration_s = float(summary.get("duration", 0))
    if distance_m <= 0 or duration_s <= 0 or not coordinates:
        raise PredictionError("경로 거리 또는 예상 시간을 가져오지 못했습니다.")
    return distance_m / 1000, duration_s / 60, coordinates


def _current_temperature(client: httpx.Client, destination: Location, fallback: float) -> tuple[float, str | None]:
    try:
        response = client.get(
            OPEN_METEO_URL,
            params={
                "latitude": destination.latitude,
                "longitude": destination.longitude,
                "current": "temperature_2m",
                "timezone": "auto",
            },
        )
        response.raise_for_status()
        temperature = response.json().get("current", {}).get("temperature_2m")
        if temperature is None:
            raise ValueError("temperature_2m is missing")
        return float(temperature), None
    except (httpx.HTTPError, ValueError, TypeError):
        return fallback, "현재 날씨를 가져오지 못해 학습 데이터의 대표 온도를 사용했습니다."


def _map_html(start: Location, destination: Location, coordinates: list[list[float]]) -> str:
    route_latlon = [(latitude, longitude) for longitude, latitude, *_ in coordinates]
    route_map = folium.Map(location=route_latlon[len(route_latlon) // 2], zoom_start=8, control_scale=True)
    folium.TileLayer("CartoDB positron", name="밝은 지도").add_to(route_map)
    folium.PolyLine(route_latlon, color="#00786f", weight=6, opacity=0.9).add_to(route_map)
    folium.Marker(
        [start.latitude, start.longitude],
        tooltip=html.escape(f"출발: {start.label}"),
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(route_map)
    folium.Marker(
        [destination.latitude, destination.longitude],
        tooltip=html.escape(f"도착: {destination.label}"),
        icon=folium.Icon(color="red", icon="flag"),
    ).add_to(route_map)
    route_map.fit_bounds(route_latlon, padding=(24, 24))
    return route_map.get_root().render()


@lru_cache(maxsize=1)
def _model_context() -> tuple[Any, pd.DataFrame, float]:
    frame = load_ev_data(DATA_PATH)
    train, validation, _ = split_ev_data(frame)
    model = train_regressor(train, BASIC_FEATURES)
    model_mae = evaluate_regressor(model, validation, BASIC_FEATURES)["mae"]
    return model, train, model_mae


def predict_route_energy(start_query: str, destination_query: str, mode: str) -> RoutePrediction:
    """주소 두 개와 추천 모드에서 지도와 에너지 추천 결과를 만든다."""
    start_query, destination_query = start_query.strip(), destination_query.strip()
    if not start_query or not destination_query:
        raise PredictionError("출발지와 목적지를 모두 입력해 주세요.")
    if mode not in MODE_LIMITS:
        raise PredictionError("지원하지 않는 추천 모드입니다.")

    model, train, model_mae = _model_context()
    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        key = _api_key()
        start = _geocode(client, start_query, key, "출발지")
        destination = _geocode(client, destination_query, key, "목적지")
        distance_km, duration_minutes, coordinates = _route(client, start, destination, key)
        temperature_c, weather_notice = _current_temperature(
            client, destination, float(train["ambient_temp_C"].median())
        )

    base_speed = distance_km / duration_minutes * 60
    recommendation = recommend_trip(
        model=model,
        train_frame=train,
        trip_distance_km=distance_km,
        ambient_temp_C=temperature_c,
        base_trip_minutes=duration_minutes,
        profile={
            "speed_kmh": base_speed,
            "hvac_power_kw": float(train["hvac_power_kw"].median()),
            "driving_style_index": float(train["driving_style_index"].median()),
            "tire_pressure_bar": float(train["tire_pressure_bar"].median()),
        },
        mode=mode,
        manufacturer_tire_range=DEFAULT_TIRE_RANGE,
        model_mae=model_mae,
        features=BASIC_FEATURES,
    )
    return RoutePrediction(
        start=start,
        destination=destination,
        distance_km=distance_km,
        duration_minutes=duration_minutes,
        temperature_c=temperature_c,
        weather_notice=weather_notice,
        map_html=_map_html(start, destination, coordinates),
        recommendation=recommendation,
    )
