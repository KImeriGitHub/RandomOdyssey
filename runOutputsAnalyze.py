from pathlib import Path
from typing import Iterable, List, Tuple
import datetime
import logging
import re
import polars as pl

from src.predictionModule.LoadupSamples import LoadupSamples
from src.predictionModule.ModelAnalyzer import ModelAnalyzer
import treetimeParams

# =============================
# Configuration
# =============================
STOCK_GROUP = "group_finanTo2011"
STOCK_GROUP_SHORT = "_".join(STOCK_GROUP.split("_")[1:])
DEFAULT_OUTPUTS_DIR = Path("outputs")
DEFAULT_GLOBAL_START_DATE = datetime.date(2024, 1, 1)

# Filenames look like:
#   output_prediction_TreeTime_<group>_<ddMonYY>_<HHMM>[__completed].parquet
RAW_FILE_RE = re.compile(
    r"^output_prediction_TreeTime_(?P<group>.+?)_(?P<date>\d{1,2}[A-Za-z]{3}\d{2})_(?P<time>\d{4})(?:\.parquet)?$",
    re.IGNORECASE,
)
DONE_FILE_RE = re.compile(
    r"^output_prediction_TreeTime_(?P<group>.+?)_(?P<date>\d{1,2}[A-Za-z]{3}\d{2})_(?P<time>\d{4})__completed\.parquet$",
    re.IGNORECASE,
)

# =============================
# Logging
# =============================

def _setup_logger(group_short: str) -> logging.Logger:
    """Configure a file logger and return it."""
    formatted_date = datetime.datetime.now().strftime("%d%b%y_%H%M").lower()
    log_file = f"logs/output_PredAnalyze_{group_short}_{formatted_date}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Analyzing %s", group_short)
    return logger

LOGGER = _setup_logger(STOCK_GROUP_SHORT)

# =============================
# Filename helpers
# =============================

def completed_filename(group: str, datestr: str, timestr: str) -> str:
    """Return the canonical completed parquet filename for a given triplet."""
    return f"output_prediction_TreeTime_{group}_{datestr}_{timestr}__completed.parquet"

# =============================
# Discovery & IO
# =============================

def list_files_with_meta(directory: Path, pattern: re.Pattern) -> Tuple[List[Path], pl.DataFrame]:
    """Return matching files and a Polars metadata frame extracted from filenames.

    The metadata contains: group, datestr, timestr, path, date (pl.Date), time (pl.Time).
    Sorted by date, time, then path for deterministic processing.
    """
    files = sorted((p for p in directory.iterdir() if p.is_file()), key=lambda p: p.name.lower())
    rows, paths = [], []

    for p in files:
        m = pattern.match(p.name)
        if not m:
            continue
        paths.append(p)
        rows.append(
            {
                "group": m["group"],
                "datestr": m["date"],
                "timestr": m["time"],
                "path": str(p),
            }
        )

    if not rows:
        empty = pl.DataFrame(schema={
            "group": pl.Utf8, "datestr": pl.Utf8, "timestr": pl.Utf8, "path": pl.Utf8,
        })
        return [], empty

    meta = (
        pl.DataFrame(rows)
        .with_columns(
            date=pl.col("datestr").str.strptime(pl.Date, format="%d%b%y", strict=False),
            time=pl.col("timestr").str.strptime(pl.Time, format="%H%M"),
        )
        .sort(["date", "time", "path"])  # ensure consistent, chronological order
    )

    # Align sorted file list with meta order
    ordered_paths = [Path(p) for p in meta["path"].to_list()]
    return ordered_paths, meta


def read_tables(paths: Iterable[Path]) -> List[pl.DataFrame]:
    """Read a list of parquet files into Polars DataFrames."""
    tables: List[pl.DataFrame] = []
    for p in paths:
        try:
            tables.append(pl.read_parquet(p))
        except Exception as exc:
            LOGGER.warning("Failed to read %s: %s", p, exc)
    return tables

# =============================
# Feature joining
# =============================

def _build_meta_test_from_tables(
    tables: List[pl.DataFrame],
    global_start_date: datetime.date,
    params,
    stock_group: str,
) -> pl.DataFrame:
    """Build meta_test (date, ticker, target_ratio) via LoadupSamples using dates in provided tables."""
    test_dates = sorted({d for tbl in tables for d in tbl.select(pl.col("date")).to_series()})
    ls = LoadupSamples(
        train_start_date=global_start_date,
        test_dates=test_dates,
        group=stock_group,
        group_type="Tree",
        params=params,
    )
    ls.load_samples()
    return ls.meta_pl_test.select(["date", "ticker", "target_ratio"])


def _join_with_meta(table: pl.DataFrame, meta_test: pl.DataFrame) -> pl.DataFrame:
    """Left-join a prediction table with meta_test on (date, ticker)."""
    return table.join(meta_test, on=["date", "ticker"], how="left")


# =============================
# Writing completed files
# =============================

def write_completed_files(
    raw_paths: List[Path],
    raw_rows: List[dict],
    raw_tables: List[pl.DataFrame],
    meta_test: pl.DataFrame,
    out_dir: Path,
) -> List[Path]:
    """Join raw tables with meta_test and write ONLY those with no nulls in target_ratio.

    If the joined table has nulls in target_ratio, skip writing a completed file for that input.
    Returns the list of written (or already-existing) completed paths.
    """
    written: List[Path] = []

    for p, row, tbl in zip(raw_paths, raw_rows, raw_tables):
        out_name = completed_filename(row["group"], row["datestr"], row["timestr"])
        out_path = out_dir / out_name

        if out_path.exists():
            LOGGER.info("Completed exists, skipping write: %s", out_path)
            written.append(out_path)
            continue

        joined = _join_with_meta(tbl, meta_test)
        has_nulls = joined.select(pl.col("target_ratio").is_null().any()).item()
        if has_nulls:
            LOGGER.warning("Skipping completed write due to null values: %s", p)
            continue

        joined.write_parquet(out_path)
        LOGGER.info("Wrote completed file: %s", out_path)
        written.append(out_path)

    return written

# =============================
# Analysis
# =============================

def analyze_tables(tables: List[pl.DataFrame], meta: pl.DataFrame, expected_group_short: str) -> None:
    """Run ModelAnalyzer across completed tables.

    Note: The filename-derived `group` is compared to the *short* group name (postfix after first '_').
    This mirrors the original behavior.
    """
    results: List[dict] = []

    for i, tbl in enumerate(tables):
        # Safety checks
        if tbl is None or tbl.select("target_ratio").to_series().has_nulls():
            LOGGER.warning("Table %s could not be analyzed due to null values.", meta["path"].item(i))
            continue

        if meta["group"].item(i) != expected_group_short:
            LOGGER.warning("Group mismatch. Skipping table %s", meta["path"].item(i))
            continue

        LOGGER.info("Analyzing table %d/%d", i + 1, len(tables))
        LOGGER.info("Date: %s, Time: %s", meta["date"].item(i), meta["time"].item(i))
        LOGGER.info("Path: %s", meta["path"].item(i))

        ModelAnalyzer.log_test_result_overall(tbl, last_col="target_ratio")

        results.append(
            {
                "end_train_date": tbl.select("date").min().item(),
                "end_test_date": tbl.select("date").max().item(),
                "table": tbl,
            }
        )

    if results:
        ModelAnalyzer.log_test_result_multiple([r["table"] for r in results], last_col="target_ratio")
    else:
        LOGGER.warning("No valid results found for analysis.")

# =============================
# Orchestrator
# =============================

def run_pipeline(
    params: dict,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
    global_start_date: datetime.date = DEFAULT_GLOBAL_START_DATE,
    stock_group: str = STOCK_GROUP,
    stock_group_short: str = STOCK_GROUP_SHORT,
) -> None:
    """End-to-end pipeline.

    1) Discover completed and raw output files.
    2) For raw files missing a completed counterpart, build meta and write completed files (if no nulls).
    3) Load completed files only and analyze them with ModelAnalyzer.
    """
    # 1) Discover
    done_paths, done_meta = list_files_with_meta(outputs_dir, DONE_FILE_RE)
    raw_paths, raw_meta = list_files_with_meta(outputs_dir, RAW_FILE_RE)

    # Compute the set of intended completed targets (for info/debug)
    completed_targets = set()
    if len(raw_meta) > 0:
        for i in range(len(raw_paths)):
            row = raw_meta.row(i, named=True)
            completed_targets.add((outputs_dir / completed_filename(row["group"], row["datestr"], row["timestr"])) .resolve())

    existing_completed = {p.resolve() for p in done_paths}

    # 2) Determine which raw files still need processing
    raw_to_process_idx: List[int] = []
    for i in range(len(raw_paths)):
        row = raw_meta.row(i, named=True)
        candidate = (outputs_dir / completed_filename(row["group"], row["datestr"], row["timestr"])) .resolve()
        if candidate not in existing_completed:
            raw_to_process_idx.append(i)

    # If there are raw files to process, compute meta_test and write completed files
    if raw_to_process_idx:
        tables_for_dates: List[pl.DataFrame] = []
        if done_paths:
            tables_for_dates.extend(read_tables(done_paths))
        tables_for_dates.extend(read_tables([raw_paths[i] for i in raw_to_process_idx]))

        meta_test = _build_meta_test_from_tables(
            tables_for_dates,
            global_start_date=global_start_date,
            params=params,
            stock_group=stock_group,
        )

        raw_paths_subset = [raw_paths[i] for i in raw_to_process_idx]
        raw_rows_subset = [raw_meta.row(i, named=True) for i in raw_to_process_idx]
        raw_tables_subset = read_tables(raw_paths_subset)

        write_completed_files(
            raw_paths=raw_paths_subset,
            raw_rows=raw_rows_subset,
            raw_tables=raw_tables_subset,
            meta_test=meta_test,
            out_dir=outputs_dir,
        )

        # Refresh completed listings after writes
        done_paths, done_meta = list_files_with_meta(outputs_dir, DONE_FILE_RE)

    # 3) Load completed and analyze
    if not done_paths:
        LOGGER.warning("No completed files found and nothing processed. Nothing to analyze.")
        return

    completed_tables = read_tables(done_paths)
    analyze_tables(completed_tables, done_meta, expected_group_short=stock_group_short)


# =============================
# Entry point
# =============================
if __name__ == "__main__":
    params = treetimeParams.params
    run_pipeline(
        params=params,
        outputs_dir=DEFAULT_OUTPUTS_DIR,
        global_start_date=DEFAULT_GLOBAL_START_DATE,
        stock_group=STOCK_GROUP,
        stock_group_short=STOCK_GROUP_SHORT,
    )