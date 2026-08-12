import pandas as pd
from firerisk.grid import to_cell_id, to_cell_id_vec, cell_center, neighbours

def test_round_trip():
    cid = to_cell_id(36.73, 4.05)
    assert cid == "367_41"
    lat, lon = cell_center(cid)
    assert abs(lat - 36.7) < 1e-9
    assert abs(lon - 4.1) < 1e-9

def test_negative_longitude():
    assert to_cell_id(34.02, -2.46) == "340_-25"
    lat, lon = cell_center("340_-25")
    assert abs(lat - 34.0) < 1e-9
    assert abs(lon - (-2.5)) < 1e-9

def test_boundary_is_half_up_not_bankers():
    # 36.75 -> 367.5 must go UP to 368; Python round() would give 368,
    # but 36.65 -> 366.5 would give 366 under banker's rounding. Both must round up.
    assert to_cell_id(36.75, 0.0).split("_")[0] == "368"
    assert to_cell_id(36.65, 0.0).split("_")[0] == "367"

def test_negative_boundary_is_half_up():
    # -36.75 -> -367.5 must round UP (toward +inf) to -367, matching the
    # same half-up convention as the positive ties above.
    assert to_cell_id(-36.75, 0.0).split("_")[0] == "-367"

def test_vectorised_matches_scalar():
    lat = pd.Series([36.73, 34.02, 36.75])
    lon = pd.Series([4.05, -2.46, 0.0])
    got = to_cell_id_vec(lat, lon)
    exp = [to_cell_id(a, b) for a, b in zip(lat, lon)]
    assert list(got) == exp

def test_neighbours():
    n = neighbours("367_41")
    assert len(n) == 8
    assert "367_41" not in n
    assert "366_40" in n and "368_42" in n and "367_40" in n
