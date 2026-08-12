import datetime as dt

from firerisk.firms import season_chunk_starts, load_detections, qualifying

RAW = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight,type\n"
    "36.73,4.05,305.0,0.5,0.6,2023-07-20,46,N,VIIRS,n,2,294.0,1.9,N,0\n"
    "36.75,4.06,306.0,0.5,0.6,2023-07-20,47,N,VIIRS,l,2,294.0,2.0,N,0\n"
    "35.10,5.00,320.0,0.5,0.6,2023-07-21,50,N,VIIRS,h,2,300.0,9.0,D,0\n"
    "36.00,3.00,330.0,0.5,0.6,2023-07-21,51,N,VIIRS,h,2,310.0,50.0,D,2\n"
    "36.10,3.10,330.0,0.5,0.6,2023-07-22,52,N,VIIRS,n,2,310.0,50.0,D,3\n"
)


def test_season_chunk_starts_covers_season_with_day_range_5():
    starts = season_chunk_starts(2023, "06-01", "10-31", 5)
    assert starts[0] == dt.date(2023, 6, 1)
    assert starts[-1] <= dt.date(2023, 10, 31)
    assert all((b - a).days == 5 for a, b in zip(starts, starts[1:]))
    assert len(starts) == 31


def test_load_detections_filters_type_and_keeps_all_confidences(tmp_path):
    d = tmp_path / "raw" / "firms" / "2023"
    d.mkdir(parents=True)
    (d / "2023-07-20.csv").write_text(RAW)

    class Cfg:
        data_dir = tmp_path
        resolution = 0.1
        firms_confidence = ["n", "h"]
        year_start, year_end = 2023, 2023

        @property
        def years(self):
            return [2023]

    det = load_detections(Cfg())
    # type 2 and 3 dropped; all three type-0 rows kept regardless of confidence
    assert len(det) == 3
    assert set(det.confidence) == {"n", "l", "h"}
    assert set(det.cell_id) == {"367_41", "368_41", "351_50"}
    assert det.date.dtype.kind == "M"

    qual = qualifying(det, Cfg())
    assert len(qual) == 2
    assert "l" not in set(qual.confidence)
