TARGET = "energy_consumption_kwhper100km"

BASIC_FEATURES = [
    "speed_kmh",
    "ambient_temp_C",
    "hvac_power_kw",
    "driving_style_index",
    "tire_pressure_bar",
    "trip_distance_km",
]

EXPANDED_FEATURES = [
    *BASIC_FEATURES,
    "payload_kg",
    "battery_temp_C",
]

EXCLUDED_FEATURES = ["road_grade_pct"]
REQUIRED_COLUMNS = [*EXPANDED_FEATURES, *EXCLUDED_FEATURES, TARGET]


def validate_ev_data(frame):
    """학습용 데이터가 합의된 열과 결측치 조건을 만족하는지 확인한다."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"필수 열이 없습니다: {', '.join(missing_columns)}")

    null_columns = frame[REQUIRED_COLUMNS].columns[frame[REQUIRED_COLUMNS].isna().any()]
    if len(null_columns):
        raise ValueError(f"결측치가 있는 열: {', '.join(null_columns)}")


def load_ev_data(path):
    """CSV를 읽고 모델 학습에 사용할 수 있는지 검증한다."""
    import pandas as pd

    frame = pd.read_csv(path)
    validate_ev_data(frame)
    return frame


def split_ev_data(frame, random_state=42):
    """데이터를 훈련 70%, 검증 15%, 테스트 15%로 나눈다."""
    from sklearn.model_selection import train_test_split

    train, remainder = train_test_split(
        frame,
        test_size=0.30,
        random_state=random_state,
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=random_state,
    )
    return train, validation, test
