from src.predictionModule.LoadupSamples import LoadupSamples
import treetimeParams

import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import re
import datetime
import copy
import random

stock_group = "group_debug"
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

if __name__ == "__main__":
    tables, meta_pred = read_parquet_files()
    
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
        params=None,
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

        logger.info(f"Analyzing table {i+1}/{len(tables_joined)}")
        logger.info(f"Date: {meta_pred['date'].item(i)}, Time: {meta_pred['time'].item(i)}")

        jTable_perdate = jTable.group_by("date").agg([
            pl.col("target_ratio").mean().alias("mean_res"),
            pl.col("target_ratio").first().alias("top_res"),
            pl.col("target_ratio").count().alias("n_entries"),
            pl.col("prediction_ratio").max().alias("max_pred"),
            pl.col("prediction_ratio").mean().alias("mean_pred"),
        ])

        res_meanmean = jTable_perdate['mean_res'].mean()
        res_meanlast = jTable_perdate['mean_res'].last()
        res_topmean = jTable_perdate['top_res'].mean()
        res_toplast = jTable_perdate['top_res'].last()
        res_sum_n = jTable_perdate['n_entries'].sum()
        pred_meanmean = jTable_perdate['mean_pred'].mean()
        pred_meanlast = jTable_perdate['mean_pred'].last()
        pred_toplast = jTable_perdate['max_pred'].last()
        logger.info(f"  Overall mean P/L Ratio: {res_meanmean:.4f}")
        logger.info(f"  Overall top mean P/L Ratio: {res_topmean:.4f}")
        logger.info(f"  Overall top last P/L Ratio: {res_toplast:.4f}")
        logger.info(f"  Overall mean last P/L Ratio: {res_meanlast:.4f}")
        logger.info(f"  Overall mean prediction ratio: {pred_meanmean:.4f}")
        logger.info(f"  Overall top last prediction ratio: {pred_toplast:.4f}")
        logger.info(f"  Overall last mean prediction ratio: {pred_meanlast:.4f}")
        logger.info(f"  Overall number of entries: {res_sum_n}")

        results.append(
            {
                "end_train_date": jTable.select('date').min().item(),
                "end_test_date": jTable.select('date').max().item(),
                "res_meanmean": res_meanmean,
                "res_toplast": res_toplast,
                "res_meanlast": res_meanlast,
                "n_entries": res_sum_n,
                "pred_toplast": pred_toplast,
                "pred_meanmean": pred_meanmean,
                "pred_meanlast": pred_meanlast,
            }
        )

    results_df = pd.DataFrame(results).sort_values("end_train_date").reset_index(drop=True)
    logger.info(f"\n {results_df.to_string(index=False)}")

    logger.info(f"Mean over meanmean returns over all cutoffs: {results_df['res_meanmean'].mean()}")
    logger.info(f"Mean over toplast returns over all cutoffs: {results_df['res_toplast'].mean()}")
    logger.info(f"Mean over meanlast returns over all cutoffs: {results_df['res_meanlast'].mean()}")
    logger.info(f"Mean over meanmean predictions over all cutoffs: {results_df['pred_meanmean'].mean()}")
    logger.info(f"Mean over toplast predictions over all cutoffs: {results_df['pred_toplast'].mean()}")
    logger.info(f"Mean over meanlast predictions over all cutoffs: {results_df['pred_meanlast'].mean()}")
    logger.info(f"Total entries over all cutoffs: {results_df['n_entries'].sum()}")

    if len(results_df) > 3:
        logger.info(
            "Mean over meanmean returns filtered by 0.5 quantile prediction meanmean: "
            f"{results_df.loc[results_df['pred_meanmean'] > results_df['pred_meanmean'].quantile(0.5), 'res_meanmean'].mean()}"
        )
        logger.info(
            "Mean over meanlast returns filtered by 0.5 quantile prediction meanmean: "
            f"{results_df.loc[results_df['pred_meanmean'] > results_df['pred_meanmean'].quantile(0.5), 'res_meanlast'].mean()}"
        )
        logger.info(
            f"Mean over meanlast returns filtered by 0.5 quantile prediction meanlast: "
            f"{results_df.loc[results_df['pred_meanlast'] > results_df['pred_meanlast'].quantile(0.5), 'res_meanlast'].mean()}"
        )
        logger.info(
            f"Mean over toplast returns filtered by 0.5 quantile prediction toplast: "
            f"{results_df.loc[results_df['pred_toplast'] > results_df['pred_toplast'].quantile(0.5), 'res_toplast'].mean()}"
        )
        logger.info(
            f"Mean over toplast returns filtered by 0.5 quantile prediction meanmean: "
            f"{results_df.loc[results_df['pred_meanmean'] > results_df['pred_meanmean'].quantile(0.5), 'res_toplast'].mean()}"
        )