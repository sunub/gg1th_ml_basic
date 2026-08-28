"""전기차 소비율 예측 모델의 학습과 성능 비교."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

from .data import BASIC_FEATURES, EXPANDED_FEATURES, TARGET, load_ev_data, split_ev_data

RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 3, 5],
    "max_features": [0.7, 1.0],
}


def train_regressor(
    train: pd.DataFrame,
    features: list[str],
    random_state: int = 42,
) -> XGBRegressor:
    """지정한 특징으로 비선형 소비율 예측 모델을 학습한다."""
    model = XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train[features], train[TARGET])
    return model


def tune_random_forest(
    train: pd.DataFrame,
    features: list[str] | None = None,
    param_grid: dict[str, list[object]] | None = None,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> GridSearchCV:
    """5-fold 교차 검증 MAE로 Random Forest 설정을 탐색한다.

    테스트 데이터는 이 함수에 전달하지 않는다. 이 함수의 최적 설정은 훈련
    데이터 안에서만 선택되며, 최종 평가는 별도 테스트 데이터에서 수행한다.
    """
    selected_features = BASIC_FEATURES if features is None else features
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=random_state, n_jobs=1),
        param_grid=RANDOM_FOREST_PARAM_GRID if param_grid is None else param_grid,
        scoring="neg_mean_absolute_error",
        cv=splitter,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )
    search.fit(train[selected_features], train[TARGET])
    return search


def cv_leaderboard(search: GridSearchCV) -> pd.DataFrame:
    """교차 검증 점수를 사람이 읽을 수 있는 MAE 순위표로 바꾼다."""
    results = pd.DataFrame(search.cv_results_)
    parameter_columns = [
        column for column in results.columns if column.startswith("param_")
    ]
    ranking = results[["mean_test_score", "std_test_score", *parameter_columns]].copy()
    ranking = ranking.rename(
        columns={"mean_test_score": "mean_mae", "std_test_score": "std_mae"}
    )
    ranking["mean_mae"] = -ranking["mean_mae"]
    ranking = ranking.sort_values(["mean_mae", "std_mae"], ignore_index=True)
    return ranking


def select_stable_configuration(
    ranking: pd.DataFrame,
    mae_tolerance: float = 0.05,
) -> pd.Series:
    """비슷한 평균 MAE 후보 중 분할별 오차가 가장 안정적인 설정을 고른다."""
    if ranking.empty:
        raise ValueError("선택할 교차 검증 결과가 없습니다.")
    if mae_tolerance < 0:
        raise ValueError("MAE 허용 차이는 0 이상이어야 합니다.")

    best_mae = ranking["mean_mae"].min()
    comparable = ranking.loc[ranking["mean_mae"] <= best_mae + mae_tolerance]
    return comparable.sort_values(["std_mae", "mean_mae"], ignore_index=True).iloc[0]


def evaluate_regressor(
    model: Any,
    frame: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    """소비율 예측 결과를 사용자가 해석할 수 있는 지표로 변환한다."""
    actual = frame[TARGET]
    predicted = model.predict(frame[features])
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate_mean_baseline(
    train: pd.DataFrame,
    frame: pd.DataFrame,
) -> dict[str, float]:
    """훈련 데이터 평균만 예측하는 기준선의 성능을 계산한다."""
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(train[[TARGET]], train[TARGET])
    return evaluate_regressor(baseline, frame, [TARGET])


def select_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    """검증 MAE로 기본·확장 모델을 고르고 테스트 결과를 한 번만 계산한다."""
    baseline_metrics = evaluate_mean_baseline(train, validation)
    basic_model = train_regressor(train, BASIC_FEATURES)
    expanded_model = train_regressor(train, EXPANDED_FEATURES)
    basic_metrics = evaluate_regressor(basic_model, validation, BASIC_FEATURES)
    expanded_metrics = evaluate_regressor(
        expanded_model,
        validation,
        EXPANDED_FEATURES,
    )

    if basic_metrics["mae"] <= expanded_metrics["mae"]:
        model, features, selected_name = basic_model, BASIC_FEATURES, "기본 모델"
    else:
        model, features, selected_name = expanded_model, EXPANDED_FEATURES, "확장 모델"

    comparison = pd.DataFrame.from_dict(
        {
            "기준선": baseline_metrics,
            "기본 모델": basic_metrics,
            "확장 모델": expanded_metrics,
        },
        orient="index",
    )
    return {
        "model": model,
        "features": features,
        "selected_name": selected_name,
        "validation_metrics": comparison,
        "test_metrics": evaluate_regressor(model, test, features),
    }


def main() -> None:
    """명령줄에서 모델 성능 비교를 실행한다."""
    frame = load_ev_data(Path("datas_ml/ev_energy_consumption.csv"))
    train, validation, test = split_ev_data(frame)
    result = select_model(train, validation, test)
    print("검증 데이터 성능")
    print(result["validation_metrics"].round(3))
    print(f"\n선택 모델: {result['selected_name']}")
    print("테스트 데이터 성능")
    print(pd.Series(result["test_metrics"]).round(3))


if __name__ == "__main__":
    main()
