"""Export per-cell MOD13Q1 NDVI to Drive, one CSV per year, then download.

Aggregation happens SERVER-SIDE with reduceRegions over each cell polygon.
Pulling the imagery down to reduce locally would move hundreds of GB to
produce 850k numbers.

Years 2000-2011 are fetched for the climatological baseline only; 2012-2025
are the modelling period. Both are needed: an anomaly measured against a
normal that already contains the fires being predicted is a dampened anomaly.

    python scripts/fetch_ndvi.py --project ee-abdelmalek

Resumable. A year whose CSV is already on disk is skipped entirely, so an
interrupted run costs nothing but the years still in flight.
"""
import argparse
import io
import time
from pathlib import Path

import ee
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

HALF = 0.05          # cell half-width; universe lat/lon are cell CENTRES
SCALE = 250          # MOD13Q1 native resolution, in metres
FOLDER = "firerisk_ndvi"
POLL_SECONDS = 60


def cell_collection(universe):
    """One ee.Feature per cell, carrying cell_id, as a 0.1 degree rectangle."""
    feats = [
        ee.Feature(
            ee.Geometry.Rectangle([r.lon - HALF, r.lat - HALF,
                                   r.lon + HALF, r.lat + HALF]),
            {"cell_id": r.cell_id},
        )
        for r in universe.itertuples()
    ]
    return ee.FeatureCollection(feats)


def year_task(cells, year):
    """One export task covering every composite in one year."""
    col = (ee.ImageCollection("MODIS/061/MOD13Q1")
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))

    def reduce_img(img):
        # SummaryQA <= 1 keeps good and marginal pixels, drops snow and cloud.
        masked = img.select("NDVI").updateMask(img.select("SummaryQA").lte(1))
        stamp = img.date().format("YYYY-MM-dd")
        return (masked.reduceRegions(
                    collection=cells,
                    reducer=ee.Reducer.mean().combine(
                        ee.Reducer.count(), "", True),
                    scale=SCALE)
                .map(lambda f: f.set("start", stamp)))

    return ee.batch.Export.table.toDrive(
        collection=col.map(reduce_img).flatten(),
        description=f"ndvi_{year}",
        folder=FOLDER,
        fileNamePrefix=f"ndvi_{year}",
        fileFormat="CSV",
        selectors=["cell_id", "start", "mean", "count"],
    )


def drive_service():
    """Reuse the Earth Engine token - it already carries the Drive scope."""
    return build("drive", "v3", credentials=ee.data.get_persistent_credentials(),
                 cache_discovery=False)


def download_year(svc, year, out):
    """Pull ndvi_<year>.csv out of Drive. Returns True if it landed."""
    resp = svc.files().list(
        q=f"name contains 'ndvi_{year}' and trashed = false",
        fields="files(id, name, size)", pageSize=50,
    ).execute()
    files = [f for f in resp.get("files", []) if f["name"].endswith(".csv")]
    if not files:
        return False

    dest = out / f"ndvi_{year}.csv"
    with open(dest, "wb") as fh:
        # GEE shards very large tables; concatenate, keeping one header.
        for i, meta in enumerate(sorted(files, key=lambda f: f["name"])):
            buf = io.BytesIO()
            dl = MediaIoBaseDownload(buf, svc.files().get_media(fileId=meta["id"]))
            done = False
            while not done:
                _, done = dl.next_chunk()
            text = buf.getvalue()
            if i > 0:
                text = text.split(b"\n", 1)[1]
            fh.write(text)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--universe", default="data/interim/universe.parquet")
    ap.add_argument("--out", default="data/raw/ndvi")
    ap.add_argument("--from-year", type=int, default=2000)
    ap.add_argument("--to-year", type=int, default=2025)
    args = ap.parse_args()

    ee.Initialize(project=args.project)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    universe = pd.read_parquet(args.universe)
    print(f"{len(universe):,} cells, {args.from_year}-{args.to_year}")
    cells = cell_collection(universe)

    wanted = range(args.from_year, args.to_year + 1)
    todo = [y for y in wanted if not (out / f"ndvi_{y}.csv").exists()]
    if not todo:
        print("every year already downloaded, nothing to do")
        return

    svc = drive_service()

    # A previous run may have finished the export but not the download.
    recovered = [y for y in todo if download_year(svc, y, out)]
    for y in recovered:
        print(f"  {y} recovered from Drive")
    todo = [y for y in todo if y not in recovered]
    if not todo:
        print("all years recovered from Drive")
        return

    tasks = {}
    for year in todo:
        t = year_task(cells, year)
        t.start()
        tasks[year] = t
        print(f"  {year} submitted ({t.id})")

    print(f"\n{len(tasks)} export tasks running. Polling every {POLL_SECONDS}s.")
    pending, failed = dict(tasks), {}
    while pending:
        time.sleep(POLL_SECONDS)
        for year, t in list(pending.items()):
            state = t.status()["state"]
            if state == "COMPLETED":
                ok = download_year(svc, year, out)
                print(f"  {year} COMPLETED, downloaded={ok}", flush=True)
                del pending[year]
            elif state in ("FAILED", "CANCELLED"):
                failed[year] = t.status().get("error_message", state)
                print(f"  {year} {state}: {failed[year]}", flush=True)
                del pending[year]
        if pending:
            print(f"  ... {len(pending)} still running", flush=True)

    have = sorted(int(p.stem.split("_")[1]) for p in out.glob("ndvi_*.csv"))
    print(f"\n{len(have)} of {len(list(wanted))} years on disk")
    if failed:
        print(f"failed: {sorted(failed)} - re-run to retry just those")


if __name__ == "__main__":
    main()
