"""전기차 소비율 데이터의 품질 점검과 시각화 함수."""

import numpy as np
import pandas as pd

from .data import TARGET


def configure_plot_font() -> None:
    """macOS 환경에서 한글 차트 제목과 축을 읽을 수 있게 설정한다."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = [
        "Apple SD Gothic Neo",
        "NanumGothic",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def data_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """수치형 열의 결측치와 기초 통계를 열 단위 표로 반환한다."""
    numeric = frame.select_dtypes(include="number")
    return pd.DataFrame(
        {
            "missing_count": frame.isna().sum(),
            "min": numeric.min(),
            "median": numeric.median(),
            "mean": numeric.mean(),
            "max": numeric.max(),
            "std": numeric.std(),
        }
    )


def plot_distribution(frame: pd.DataFrame, column: str):
    """한 변수의 히스토그램과 박스플롯을 나란히 그린다."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    configure_plot_font()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=frame, x=column, kde=True, ax=axes[0])
    sns.boxplot(data=frame, x=column, ax=axes[1])
    axes[0].set_title(f"{column} 분포")
    axes[1].set_title(f"{column} 박스플롯")
    figure.tight_layout()
    return figure, axes


def plot_missing_values(frame: pd.DataFrame):
    """열별 결측치 수를 막대그래프로 그린다."""
    import matplotlib.pyplot as plt

    configure_plot_font()
    missing = frame.isna().sum().sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(10, 4))
    missing.plot.bar(ax=axis)
    axis.set_title("열별 결측치 수")
    axis.set_xlabel("열")
    axis.set_ylabel("결측치 수")
    figure.tight_layout()
    return axis


def plot_feature_relationship(frame: pd.DataFrame, feature: str):
    """특징과 소비율의 산점도 및 선형 추세를 그린다."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    configure_plot_font()
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.regplot(
        data=frame,
        x=feature,
        y=TARGET,
        scatter_kws={"alpha": 0.25, "s": 14},
        line_kws={"color": "crimson"},
        ax=axis,
    )
    axis.set_title(f"{feature}와 소비율의 관계")
    axis.set_ylabel("소비율 (kWh/100km)")
    figure.tight_layout()
    return axis


def plot_binned_boxplot(frame: pd.DataFrame, feature: str, bins: int = 8):
    """연속형 특징을 구간화하여 구간별 소비율 분포를 비교한다."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    configure_plot_font()
    working = frame[[feature, TARGET]].copy()
    working["구간"] = pd.cut(working[feature], bins=bins)
    figure, axis = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=working, x="구간", y=TARGET, ax=axis)
    axis.set_title(f"{feature} 구간별 소비율")
    axis.set_xlabel(feature)
    axis.set_ylabel("소비율 (kWh/100km)")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    return axis


def plot_speed_temperature_heatmap(
    frame: pd.DataFrame,
    min_count: int = 20,
    bins: int = 8,
):
    """표본 수가 충분한 속도·온도 구간의 평균 소비율을 히트맵으로 표시한다."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    configure_plot_font()
    working = frame[["speed_kmh", "ambient_temp_C", TARGET]].copy()
    working["속도 구간"] = pd.cut(working["speed_kmh"], bins=bins)
    working["온도 구간"] = pd.cut(working["ambient_temp_C"], bins=bins)
    grouped = working.groupby(["온도 구간", "속도 구간"], observed=True)[TARGET]
    mean_table = grouped.mean().unstack()
    count_table = grouped.count().unstack().reindex_like(mean_table)
    masked = mean_table.mask(count_table < min_count)
    labels = count_table.fillna(0).astype(int).astype(str)

    figure, axis = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        masked,
        mask=masked.isna(),
        annot=labels,
        fmt="",
        cmap="YlOrRd",
        cbar_kws={"label": "평균 소비율 (kWh/100km)"},
        ax=axis,
    )
    axis.set_title(f"속도·온도 구간별 평균 소비율 (표본 수 {min_count} 미만 제외)")
    axis.set_xlabel("속도 구간 (km/h)")
    axis.set_ylabel("온도 구간 (°C)")
    figure.tight_layout()
    return axis
