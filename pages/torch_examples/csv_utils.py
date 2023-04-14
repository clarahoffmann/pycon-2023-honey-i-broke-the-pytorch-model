"""Reformat csv files written by CSV loggers during training"""
from pathlib import Path

import pandas as pd


def format_csv(
    csv_path: str, metric_list: list[str], output_path: Path
) -> None:

    """
    Reformat csv files to be usable in plots
    """

    metrics_df = pd.read_csv(csv_path)

    df_list = []
    for metric in metric_list:
        if metric == "train_loss":
            epochs = metrics_df[~metrics_df["train_loss"].isnull()]["epoch"][
                ::3
            ]
            train_loss = metrics_df[~metrics_df["train_loss"].isnull()][
                "train_loss"
            ][::3]
            data = pd.DataFrame(
                {"epoch": epochs, "metric": train_loss, "label": "train_loss"}
            )
            df_list.append(data)
        else:
            epochs_metric = metrics_df[~metrics_df["val_loss"].isnull()][
                "epoch"
            ]
            metric_value = metrics_df[~metrics_df["val_loss"].isnull()][
                "val_loss"
            ]
            data = pd.DataFrame(
                {
                    "epoch": epochs_metric,
                    "metric": metric_value,
                    "label": metric,
                }
            )
            df_list.append(data)

    formatted_metrics_df = pd.concat(df_list)
    formatted_metrics_df.to_csv(output_path, index=False)
