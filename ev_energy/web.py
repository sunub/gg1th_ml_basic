"""지도 캔버스 위에서 EV 경로 예측을 제공하는 Gradio 화면."""
from __future__ import annotations

import base64

import folium
import gradio as gr

from .service import PredictionError, RoutePrediction, predict_route_energy

MODE_LABELS = {"fast": "빠르게", "balanced": "균형", "saver": "절약 우선"}

CSS = """
:root{--ink:#17211f;--lime:#c8ff5a;--paper:#f8fbf7;--shadow:0 20px 60px #10221c36}html,body{margin:0;background:#e3ebe5}.gradio-container{--body-background-fill:transparent;--background-fill-primary:#fff;--block-background-fill:#fff;--input-background-fill:#fff;--body-text-color:#17211f;max-width:none!important;min-height:100svh;margin:0!important;padding:0!important;background:#e3ebe5;font-family:Inter,ui-sans-serif,system-ui}.gradio-container>.main{max-width:none!important;padding:0!important}.topbar{position:fixed;z-index:20;inset:0 0 auto;height:64px;display:flex;align-items:center;justify-content:space-between;padding:0 28px;color:#fff;background:linear-gradient(#101817d9,#10181700);pointer-events:none}.brand{font-size:20px;font-weight:800;letter-spacing:-.04em}.route-state,.kicker{color:var(--lime);font:600 11px ui-monospace,monospace;letter-spacing:.12em}#map-stage{position:relative;z-index:1;width:100vw!important;min-height:100svh;margin:0!important;border:0!important;border-radius:0!important;overflow:hidden;background:#dfe8e2}#map-stage>.wrap,#map-stage>.wrap>div{height:100%}.map-empty{min-height:100svh;display:grid;place-items:center;text-align:center;color:#52625c;line-height:1.6;background:radial-gradient(circle at 65% 30%,#f5f9f4,transparent 35%),linear-gradient(130deg,#dce7df,#f7faf5)}.map-empty b{display:block;color:#17211f;font:600 12px ui-monospace,monospace;letter-spacing:.15em}.route-map-frame{display:block;width:100%;height:100svh;border:0}#command-island{position:fixed!important;z-index:15;top:82px;left:28px;right:28px;width:auto!important;margin:0!important;padding:14px!important;border:1px solid #ffffff75!important;border-radius:18px!important;background:#f8fbf7e8!important;box-shadow:var(--shadow);backdrop-filter:blur(18px)}#command-island .form,#command-island .gr-row,#command-island .wrap,#command-island .block,#command-island .styler{background:transparent!important}#command-island label,#command-island .label-wrap span{color:#53625d!important;font:600 10px ui-monospace,monospace!important;letter-spacing:.12em;text-transform:uppercase}#command-island input,#command-island [role=combobox]{min-height:42px;background:#fff!important;color:var(--ink)!important;border:1px solid #d1dbd4!important;border-radius:11px!important}.run-button,.run-button button{min-height:42px!important;border:0!important;border-radius:11px!important;background:var(--lime)!important;color:#253900!important;font-weight:800!important;transition:.18s!important}.run-button:hover{transform:translateY(-1px)}.run-button button:disabled{background:#26322e!important;color:#edf5e6!important;animation:pending 1s ease-in-out infinite}@keyframes pending{50%{opacity:.62;transform:scale(.98)}}#status-island{position:fixed;z-index:16;top:170px;left:28px;width:auto!important;margin:0!important;pointer-events:none}.status{display:inline-flex;align-items:center;gap:8px;padding:9px 12px;border-radius:999px;background:#15211ddd;color:#eff6ef;font:600 11px ui-monospace,monospace}.status.loading:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--lime);animation:pulse 1s infinite}@keyframes pulse{50%{transform:scale(.4);opacity:.4}}.result{position:fixed;z-index:15;right:28px;top:198px;width:min(360px,calc(100vw - 56px));padding:20px;border:1px solid #ffffff70;border-radius:20px;background:#f8fbf7ed;box-shadow:var(--shadow);backdrop-filter:blur(18px);animation:rise .35s ease}.result h2{margin:5px 0 18px;font-size:23px;letter-spacing:-.04em}.energy{padding:15px;border-radius:14px;background:#192823;color:#fff}.energy .kicker{color:#b8cbbd}.energy strong{display:block;margin-top:5px;color:var(--lime);font-size:33px;letter-spacing:-.06em}.metric{display:flex;justify-content:space-between;padding:13px 0;border-bottom:1px solid #d8e0da}.metric span{color:#5a6962;font-size:13px}.saving{margin-top:15px;color:#31510a;font-size:13px;line-height:1.55}.tray{position:fixed;z-index:15;bottom:24px;left:28px;right:28px;display:grid;grid-template-columns:1fr auto;gap:20px;align-items:end;padding:16px 18px;border:1px solid #ffffff70;border-radius:18px;background:#f8fbf7ed;box-shadow:var(--shadow);backdrop-filter:blur(18px);animation:rise .42s ease}.tray p{margin:0 0 9px;color:#62726b;font:600 10px ui-monospace,monospace;letter-spacing:.14em}.tray table{width:100%;border-collapse:collapse;font-size:12px}.tray th{color:#65746e;text-align:left}.tray td{padding-top:7px}.tray tr:last-child td{color:#345b04;font-weight:700}.controls{min-width:174px;padding-left:18px;border-left:1px solid #d7e0d9;color:#46574f;font-size:12px;line-height:1.8}.notice{position:fixed;z-index:16;right:28px;top:198px;width:min(360px,calc(100vw - 56px));padding:14px;border-radius:14px;background:#4b3213f0;color:#fff5dc;box-shadow:var(--shadow)}.gradio-container footer{display:none!important}@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}@media(max-width:720px){#command-island{top:68px;left:12px;right:12px}#command-island .gr-row{flex-wrap:wrap!important}#status-island{top:235px;left:12px}.result{top:auto;bottom:12px;left:12px;right:12px;width:auto}.tray{display:none}.topbar{padding:0 14px}}
"""
CSS += """
#command-island { background:#14201c !important; border-color:#718078 !important; }
#command-island label, #command-island .label-wrap span { color:#d7e1da !important; }
#command-island input, #command-island [role='combobox'] { background:#ffffff !important; color:#17211f !important; }
#command-island input::placeholder { color:#66736e !important; opacity:1 !important; }
#command-island input:focus, #command-island textarea:focus, #command-island [role='combobox']:focus { color:#17211f !important; caret-color:#17211f !important; background:#ffffff !important; border-color:#c8ff5a !important; box-shadow:0 0 0 3px #c8ff5a66 !important; }
#command-island, #command-island > .wrap, #command-island .form, #command-island .gr-row { min-height:78px !important; height:auto !important; overflow:visible !important; }
#command-island .gr-row { display:flex !important; align-items:end !important; visibility:visible !important; opacity:1 !important; }
"""

JS = """
document.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.isComposing || !event.target.closest("#command-island")) return;
  event.preventDefault();
  document.querySelector("#command-island .run-button:not([disabled]), #command-island .run-button button:not([disabled])")?.click();
});
"""


def _map_frame(map_html: str) -> str:
    encoded = base64.b64encode(map_html.encode("utf-8")).decode("ascii")
    return f'<iframe class="route-map-frame" title="예측 경로 지도" src="data:text/html;charset=utf-8;base64,{encoded}"></iframe>'


def _initial_map() -> str:
    """예측 전에도 한국 전체를 표시하는 지도 캔버스를 만든다."""
    map_view = folium.Map(location=[36.3, 127.8], zoom_start=7, tiles="CartoDB positron", control_scale=True)
    return _map_frame(map_view.get_root().render())


def _result(prediction: RoutePrediction) -> str:
    r = prediction.recommendation
    base_speed = prediction.distance_km / prediction.duration_minutes * 60
    confidence = "절감 효과가 예측 오차보다 큽니다." if r["saving_is_confident"] else "절감 효과가 예측 오차 범위 안에 있습니다."
    return f"""<aside class='result'><div class='kicker' style='color:#62726b'>PREDICTED TRIP</div><h2>운행 에너지</h2><div class='energy'><div class='kicker'>RECOMMENDED ENERGY</div><strong>{r['recommended_total_energy_kwh']:.1f} kWh</strong></div><div class='metric'><span>총 경로 거리</span><b>{prediction.distance_km:.1f} km</b></div><div class='metric'><span>목적지 현재 온도</span><b>{prediction.temperature_c:.1f} °C</b></div><div class='saving'><b>{confidence}</b><br/>기준 계획 대비 {r['saving_kwh']:.1f} kWh, {r['saving_percent']:.1f}% 절감 예상</div></aside><section class='tray'><div><p>BASELINE / RECOMMENDED PLAN</p><table><thead><tr><th>계획</th><th>평균 속도</th><th>총 소비량</th><th>예상 시간</th></tr></thead><tbody><tr><td>기준</td><td>{base_speed:.0f} km/h</td><td>{r['current_total_energy_kwh']:.1f} kWh</td><td>{prediction.duration_minutes:.0f}분</td></tr><tr><td>추천</td><td>{r['speed_kmh']:.0f} km/h</td><td>{r['recommended_total_energy_kwh']:.1f} kWh</td><td>{r['estimated_trip_minutes']:.0f}분</td></tr></tbody></table></div><div class='controls'><b>추천 조건</b><br/>HVAC {r['hvac_power_kw']:.1f} kW<br/>운전 성향 {r['driving_style_index']:.2f}<br/>공기압 {r['tire_pressure_bar']:.2f} bar<br/>시간 변화 {r['additional_minutes']:+.0f}분</div></section>"""


def _predict(start: str, destination: str, mode: str):
    yield "<div class='status loading'>경로와 에너지 모델을 계산하고 있습니다</div>", gr.skip(), gr.skip(), gr.update(value="예측 계산 중", interactive=False)
    try:
        prediction = predict_route_energy(start, destination, mode)
        yield "<div class='status'>● 추천 경로가 준비되었습니다</div>", _map_frame(prediction.map_html), _result(prediction), gr.update(value="다시 예측", interactive=True)
    except PredictionError as error:
        yield f"<div class='notice'>{error}</div>", gr.skip(), gr.skip(), gr.update(value="예측 실행", interactive=True)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="EV Energy Predictor") as demo:
        gr.HTML("<header class='topbar'><div class='brand'>EV Energy Predictor</div><div class='route-state'>● LIVE ROUTING</div></header>")
        map_output = gr.HTML(_initial_map(), elem_id="map-stage")
        with gr.Row(elem_id="command-island"):
            start = gr.Textbox(label="출발지", placeholder="예: 서울특별시 중구 세종대로 110", scale=4)
            destination = gr.Textbox(label="목적지", placeholder="예: 부산광역시 해운대구", scale=4)
            mode = gr.Dropdown(choices=[(label, key) for key, label in MODE_LABELS.items()], value="balanced", label="추천 모드", scale=2)
            button = gr.Button("예측 실행", variant="primary", elem_classes="run-button", scale=2)
        status = gr.HTML("<div class='status'>● 출발지와 목적지를 입력해 주세요</div>", elem_id="status-island")
        result_output = gr.HTML()
        prediction_inputs = [start, destination, mode]
        prediction_outputs = [status, map_output, result_output, button]
        button.click(_predict, prediction_inputs, prediction_outputs, concurrency_limit=1, show_progress="hidden")
        gr.on(
            triggers=[start.submit, destination.submit],
            fn=_predict,
            inputs=prediction_inputs,
            outputs=prediction_outputs,
            concurrency_limit=1,
            show_progress="hidden",
        )
    return demo
