import pandas as pd

from firerisk.weather import chunk_cells, parse_response, DAILY_VARS

DAILY = {
    "time": ["2023-06-01", "2023-06-02"],
    "temperature_2m_max": [30.0, 31.0],
    "temperature_2m_min": [18.0, 19.0],
    "temperature_2m_mean": [24.0, 25.0],
    "relative_humidity_2m_min": [30, 25],
    "relative_humidity_2m_mean": [55, 50],
    "wind_speed_10m_max": [12.0, 15.0],
    "wind_speed_10m_mean": [6.0, 7.0],
    "precipitation_sum": [0.0, 2.5],
    "vapour_pressure_deficit_max": [4.3, 5.6],
}


def test_chunking_respects_size_limit():
    uni = pd.DataFrame({"cell_id": [str(i) for i in range(325)],
                        "lat": [36.0] * 325, "lon": [4.0] * 325})
    chunks = chunk_cells(uni, 150)
    assert [len(c) for c in chunks] == [150, 150, 25]
    # 200 locations per call is the measured API hard limit (500 -> HTTP 414)
    assert all(len(c) <= 200 for c in chunks)


def test_parse_response_maps_results_to_cells_in_order():
    cells = pd.DataFrame({"cell_id": ["A", "B"], "lat": [36.7, 36.8], "lon": [4.1, 4.2]})
    payload = [{"daily": DAILY}, {"daily": DAILY}]
    df = parse_response(payload, cells)
    assert len(df) == 4
    assert set(df.cell_id) == {"A", "B"}
    assert df.date.dtype.kind == "M"
    assert df.loc[df.cell_id == "A", "precip_sum"].tolist() == [0.0, 2.5]
    assert df.loc[df.cell_id == "A", "vpd_max"].tolist() == [4.3, 5.6]


def test_parse_response_handles_single_location_dict():
    cells = pd.DataFrame({"cell_id": ["A"], "lat": [36.7], "lon": [4.1]})
    single = {k: v[:1] for k, v in DAILY.items()}
    df = parse_response({"daily": single}, cells)
    assert len(df) == 1 and df.cell_id.iloc[0] == "A"


def test_parse_response_output_columns_are_stable():
    cells = pd.DataFrame({"cell_id": ["A"], "lat": [36.7], "lon": [4.1]})
    df = parse_response({"daily": DAILY}, cells)
    assert list(df.columns) == [
        "cell_id", "date", "temp_max", "temp_min", "temp_mean",
        "rh_min", "rh_mean", "wind_max", "wind_mean", "precip_sum", "vpd_max",
    ]


def test_daily_vars_are_the_documented_set():
    assert DAILY_VARS == [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "relative_humidity_2m_min", "relative_humidity_2m_mean",
        "wind_speed_10m_max", "wind_speed_10m_mean", "precipitation_sum",
        "vapour_pressure_deficit_max",
    ]
