"""One cell, one year. Verify every MOD13Q1 assumption before bulk export.

The NDVI work rests on claims about a catalogue we have not touched: the band
names, the 0.0001 scale factor, whether system:time_start is the start or the
middle of the 16-day window, and how many pixels land in a 0.1 degree cell.
Each of those, wrong, produces plausible numbers and a silently broken feature.

    python scripts/gee_spike.py --project ee-yourname
"""
import argparse

import ee
import pandas as pd

HALF = 0.05  # cell half-width; universe lat/lon are cell CENTRES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True, help="your GEE cloud project id")
    ap.add_argument("--lat", type=float, default=36.5)
    ap.add_argument("--lon", type=float, default=3.1)
    ap.add_argument("--year", type=int, default=2021)
    args = ap.parse_args()

    ee.Initialize(project=args.project)
    print(f"initialised on {args.project}\n")

    cell = ee.Geometry.Rectangle([
        args.lon - HALF, args.lat - HALF, args.lon + HALF, args.lat + HALF,
    ])
    col = (ee.ImageCollection("MODIS/061/MOD13Q1")
           .filterDate(f"{args.year}-01-01", f"{args.year + 1}-01-01"))

    print("BAND NAMES (need NDVI and SummaryQA):")
    print(" ", ee.Image(col.first()).bandNames().getInfo(), "\n")
    print(f"COMPOSITES IN {args.year}: {col.size().getInfo()}   (expect 23)\n")

    def reduce_one(img):
        # SummaryQA <= 1 keeps good and marginal pixels, drops snow and cloud.
        masked = img.select("NDVI").updateMask(img.select("SummaryQA").lte(1))
        stats = masked.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.count(), "", True),
            geometry=cell, scale=250, maxPixels=1e9,
        )
        return ee.Feature(None, {
            "start": img.date().format("YYYY-MM-dd"),
            "ndvi_raw": stats.get("NDVI_mean"),
            "n_pixels": stats.get("NDVI_count"),
        })

    rows = col.map(reduce_one).getInfo()["features"]
    d = pd.DataFrame([r["properties"] for r in rows])
    d["start"] = pd.to_datetime(d["start"])
    d["ndvi"] = d["ndvi_raw"] * 0.0001
    d = d.sort_values("start").reset_index(drop=True)

    print("FIRST 5 COMPOSITES:")
    print(d.head().to_string(index=False), "\n")

    print("VERIFICATION")
    doys = d.start.dt.dayofyear.tolist()
    print(f"  start DOYs      {doys[:5]} ...        expect 1, 17, 33, 49, 65")
    gaps = sorted(d.start.diff().dt.days.dropna().unique())
    print(f"  gaps in days    {gaps}        expect [16] plus a short last one")
    print(f"  scaled NDVI     {d.ndvi.min():.3f} .. {d.ndvi.max():.3f}"
          f"        expect roughly -0.2 .. 0.9")
    print(f"  pixels per cell {d.n_pixels.median():.0f}"
          f"            expect ~1,400-1,700, NOT ~1")
    spring = d[d.start.dt.month.isin([3, 4])].ndvi.mean()
    summer = d[d.start.dt.month.isin([7, 8])].ndvi.mean()
    print(f"  spring {spring:.3f} vs summer {summer:.3f}"
          f"     expect spring HIGHER - vegetation cures by August")


if __name__ == "__main__":
    main()
