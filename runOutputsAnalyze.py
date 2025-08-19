from src.predictionModule.LoadupSamples import LoadupSamples
from src.predictionModule.ModelAnalyzer import ModelAnalyzer
import treetimeParams

import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import re
import datetime

stock_group = "group_finanTo2011"
stock_group_short = '_'.join(stock_group.split('_')[1:])

import logging
formatted_date = datetime.datetime.now().strftime("%d%b%y_%H%M").lower()
logging.basicConfig(
    filename=f'logs/output_PredAnalyze_{stock_group_short}_{formatted_date}.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)
logger = logging.getLogger(__name__)

logger.info(f"Starting Analyzing {stock_group_short}")


def read_parquet_files() -> tuple[list[pl.DataFrame], pl.DataFrame]:
    pat = re.compile(
        r"^output_prediction_TreeTime_(?P<group>.+?)_(?P<date>\d{1,2}[A-Za-z]{3}\d{2})_(?P<time>\d{4})(?:\.parquet)?$",
        re.IGNORECASE,
    )

    tables = []
    rows = []

    for p in Path("outputs").iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        tables.append(pl.read_parquet(p))
        rows.append({"group": m["group"], "date": m["date"], "time": m["time"], "path": str(p)})

    meta_df = pl.DataFrame(rows).with_columns(
        pl.col("date").str.strptime(pl.Date, format="%d%b%y", strict=False),
        pl.col("time").str.strptime(pl.Time, format="%H%M"),
    )

    return tables, meta_df

params = treetimeParams.params

if __name__ == "__main__":
    tables, list_info = read_parquet_files()
    
    global_start_date = datetime.date(2024, 1, 1)
    test_dates = sorted({
        date
        for table in tables
        for date in table.select(pl.col("date")).to_series()
    })
    ls = LoadupSamples(
        train_start_date=global_start_date,
        test_dates=test_dates,
        group=stock_group,
        group_type='Tree',
        params=params,
    )
    ls.load_samples()

    cols_to_add = ['target_ratio']
    meta_test = ls.meta_pl_test.select(["date", "ticker"] + cols_to_add)

    # Join on date and ticker
    tables_joined: list[pl.DataFrame] = [None] * len(tables)
    for i, table in enumerate(tables):
        table_joined = table.join(
            meta_test,
            on=["date", "ticker"],
            how="left"
        )
        tables_joined[i] = table_joined

    # Analyze Tables
    results = []
    for i,jTable in enumerate(tables_joined):
        if jTable is None or jTable.select('target_ratio').to_series().has_nulls():
            continue

        if list_info["group"].item(i) != stock_group_short:
            logger.warning("Group mismatch. Skipping")
            continue

        logger.info(f"Analyzing table {i+1}/{len(tables_joined)}")
        logger.info(f"Date: {list_info['date'].item(i)}, Time: {list_info['time'].item(i)}")

        ModelAnalyzer.log_test_result_overall(jTable, last_col="target_ratio")

        results.append({
            "end_train_date": jTable.select('date').min().item(),
            "end_test_date": jTable.select('date').max().item(),
            "table": jTable
        })
        
    if len(results) > 0:
        ModelAnalyzer.log_test_result_multiple(
            [res["table"] for res in results],
            last_col="target_ratio"
        )

    if len(results) == 0:
        logger.warning("No valid results found for analysis.")