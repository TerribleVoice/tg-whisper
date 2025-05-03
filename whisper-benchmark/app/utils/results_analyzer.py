import itertools
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("whisper-benchmark")


class ResultsAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.run_dir = self._create_run_directory()

    def _create_run_directory(self) -> Path:
        run_time = datetime.now().strftime("%d.%m.%Y_%H-%M")
        run_dir = self.output_dir / run_time
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Создана директория для результатов запуска: {run_dir}")
        return run_dir

    def save_results(self, results: Dict[str, Any], model: str, compute: str, config_name: Optional[str] = None) -> Path:
        if config_name:
            base = f"r_{config_name}"
        else:
            base = f"r_{model}_{compute}"

        result_path = self._get_unique_path(self.run_dir, base, "json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Результаты сохранены в {result_path}")
        return result_path

    def _get_unique_path(self, dir: Path, base: str, ext: str) -> Path:
        path = dir / f"{base}.{ext}"
        idx = 1
        while path.exists():
            path = dir / f"{base}_{idx}.{ext}"
            idx += 1
        return path

    def analyze_results(self, results: Dict[str, Dict[str, Any]]):
        rows = []

        for config_name, config_results in results.items():
            config = config_results["config"]

            avg_metrics = {"wer": [], "cer": [], "mer": [], "wil": []}
            diversity_vals = []
            gpu_metrics = {"max_memory_used_mb": [], "avg_utilization": []}

            for file_name, file_result in config_results["files"].items():
                for metric_name, metric_value in file_result["metrics"].items():
                    avg_metrics[metric_name].append(metric_value)

                for metric_name, metric_value in file_result.get("gpu_metrics", {}).items():
                    if metric_name in gpu_metrics:
                        gpu_metrics[metric_name].append(metric_value)

                row = {
                    "config_name": config_name,
                    "model_name": config["model_name"],
                    "file_name": file_name,
                    "duration": file_result["duration"],
                    "processing_speed": file_result["processing_speed"],
                }

                for metric_name, metric_value in file_result.get("gpu_metrics", {}).items():
                    row[f"gpu_{metric_name}"] = metric_value

                hypotheses = file_result.get("hypotheses", [])
                # diversity: среднее попарное расстояние Левенштейна между гипотезами
                if len(hypotheses) > 1:
                    pairs = list(itertools.combinations(hypotheses, 2))
                    dists = [Levenshtein.distance(a, b) for a, b in pairs]
                    diversity = np.mean(dists)
                else:
                    diversity = 0
                row["hypothesis_diversity"] = diversity
                diversity_vals.append(diversity)

                row.update(file_result["metrics"])

                for param_name, param_value in config.items():
                    if param_name != "model_name":
                        row[f"config_{param_name}"] = param_value

                rows.append(row)

            avg_row = {
                "config_name": config_name,
                "model_name": config["model_name"],
                "file_name": "AVERAGE",
                "duration": np.mean([file_result["duration"] for file_result in config_results["files"].values()]),
                "processing_speed": np.mean([file_result["processing_speed"] for file_result in config_results["files"].values()]),
            }

            for metric_name, metric_values in avg_metrics.items():
                avg_row[metric_name] = np.mean(metric_values)

            for metric_name, metric_values in gpu_metrics.items():
                if metric_values:
                    avg_row[f"gpu_{metric_name}"] = np.mean(metric_values)
                else:
                    avg_row[f"gpu_{metric_name}"] = 0

            avg_row["hypothesis_diversity"] = np.mean(diversity_vals)

            for param_name, param_value in config.items():
                if param_name != "model_name":
                    avg_row[f"config_{param_name}"] = param_value

            total_processing_time = sum(file_result.get("processing_time", 0) for _, file_result in config_results["files"].items())
            avg_row["total_processing_time"] = round(total_processing_time, 2)

            rows.append(avg_row)

        df = pd.DataFrame(rows)

        configs = [r["config"] for r in results.values()]
        base = "__".join([f"{c['name'] or 'un'}" for c in configs])
        csv_path = self._get_unique_path(self.run_dir, f"r_{base}", "csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV с результатами сохранен в {csv_path}")
        self._create_visualizations(df)

    def _make_histogram(self, df, metric, title, ylabel, filename, is_lower_better=True):
        plt.figure(figsize=(19.2, 10.8))
        plt.bar(df["config_name"], df[metric])
        plt.title(title)
        plt.xlabel("Конфигурация")
        plt.ylabel(f"{ylabel} ({'меньше - лучше' if is_lower_better else 'больше - лучше'})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots_dir = self.run_dir / "plots"
        fp = self._get_unique_path(plots_dir, filename, "png")
        plt.savefig(fp)
        plt.close()

    def _make_plot(self, df, x, y, title, xlabel, ylabel, filename, is_lower_better=True):
        plt.figure(figsize=(19.2, 10.8))
        for config_name in df["config_name"].unique():
            config_df = df[df["config_name"] == config_name].copy()
            if not config_df.empty:
                config_df = config_df.sort_values(by=x)
                plt.plot(config_df[x], config_df[y], linestyle="-", label=config_name)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(f"{ylabel} ({'меньше - лучше' if is_lower_better else 'больше - лучше'})")
        if df["config_name"].nunique() > 1:
            plt.legend()
        plt.xticks(rotation=90 if x == "file_name" else 45)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plots_dir = self.run_dir / "plots"
        fp = self._get_unique_path(plots_dir, filename, "png")
        plt.savefig(fp)
        plt.close()

    def _create_visualizations(self, df: pd.DataFrame) -> None:
        avg_df = df[df["file_name"] == "AVERAGE"].copy()
        plots_dir = self.run_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)

        self._make_histogram(avg_df, "wer", "Word Error Rate (WER) по конфигурациям", "WER", "wer", True)
        self._make_histogram(avg_df, "cer", "Character Error Rate (CER) по конфигурациям", "CER", "cer", True)
        self._make_histogram(avg_df, "mer", "Match Error Rate (MER) по конфигурациям", "MER", "mer", True)
        self._make_histogram(avg_df, "wil", "Word Information Lost (WIL) по конфигурациям", "WIL", "wil", True)
        self._make_histogram(
            avg_df,
            "processing_speed",
            "Processing Speed (отношение времени обработки к длительности аудио)",
            "Processing Speed",
            "speed",
            False,
        )
        self._make_histogram(
            avg_df,
            "hypothesis_diversity",
            "Среднее попарное расстояние Левенштейна между гипотезами (diversity)",
            "Levenshtein diversity",
            "hypothesis_diversity",
            False,
        )
        self._make_histogram(
            avg_df,
            "total_processing_time",
            "Суммарное время обработки (сек) по конфигурациям",
            "Суммарное время обработки (сек)",
            "total_processing_time",
            True,
        )

        if "gpu_max_memory_used_mb" in avg_df.columns:
            self._make_histogram(
                avg_df,
                "gpu_max_memory_used_mb",
                "Максимальное использование памяти GPU (МБ)",
                "Память GPU (МБ)",
                "gpu_memory_mb",
                True,
            )

        if "gpu_avg_utilization" in avg_df.columns:
            self._make_histogram(
                avg_df,
                "gpu_avg_utilization",
                "Средняя загрузка GPU (%)",
                "Загрузка GPU (%)",
                "gpu_utilization",
                True,
            )

        self._make_plot(
            df[df["file_name"] != "AVERAGE"],
            "duration",
            "processing_speed",
            "Зависимость скорости обработки от длительности аудио",
            "Длительность аудио (секунды)",
            "Скорость обработки",
            "duration_vs_speed",
            False,
        )
        df_files = df[df["file_name"] != "AVERAGE"].copy()
        self._make_plot(
            df_files,
            "file_name",
            "processing_speed",
            "Скорость обработки по файлам",
            "Имя файла",
            "Processing Speed",
            "speed_by_file",
            False,
        )
        df_files["accuracy"] = 1 - df_files["wer"]
        self._make_plot(
            df_files,
            "file_name",
            "accuracy",
            "Точность обработки (1 - WER) по файлам",
            "Имя файла",
            "Точность",
            "accuracy_by_file",
            False,
        )
        df_files["mer_accuracy"] = 1 - df_files["mer"]
        self._make_plot(
            df_files,
            "file_name",
            "mer_accuracy",
            "Точность обработки (1 - MER) по файлам",
            "Имя файла",
            "Точность MER",
            "mer_accuracy_by_file",
            False,
        )
        df_files["wil_accuracy"] = 1 - df_files["wil"]
        self._make_plot(
            df_files,
            "file_name",
            "wil_accuracy",
            "Точность обработки (1 - WIL) по файлам",
            "Имя файла",
            "Точность WIL",
            "wil_accuracy_by_file",
            False,
        )
        logger.info(f"Графики сохранены в директории {plots_dir}")

        self._create_summary(avg_df, plots_dir)

    def _create_summary(self, avg_df, plots_dir):
        summary_cols = ["config_name", "mer", "cer", "wil", "wer", "processing_speed", "total_processing_time"]

        if "gpu_max_memory_used_mb" in avg_df.columns:
            summary_cols.append("gpu_max_memory_used_mb")
        if "gpu_avg_utilization" in avg_df.columns:
            summary_cols.append("gpu_avg_utilization")

        summary_df = avg_df[summary_cols].copy()
        summary_df = summary_df.rename(
            columns={"processing_speed": "speed", "gpu_max_memory_used_mb": "mem_mb", "gpu_avg_utilization": "gpu_%"}
        )

        for col in summary_df.columns:
            if col != "config_name" and summary_df[col].dtype != object:
                summary_df[col] = summary_df[col].apply(lambda x: round(x, 2))

        summary_img_path = plots_dir / "summary_table.png"
        fig, ax = plt.subplots(figsize=(12, 2 + 0.5 * len(summary_df)))
        ax.axis("off")
        table = ax.table(
            cellText=summary_df.values.tolist(),
            colLabels=list(summary_df.columns),
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.2, 2)
        max_config_len = max([len(str(x)) for x in summary_df["config_name"]] + [len("config_name")])
        table.auto_set_column_width(col=list(range(len(summary_df.columns))))
        cell = table.get_celld()[(0, 0)]
        cell.set_width(0.08 + 0.01 * max(0, max_config_len - 10))

        metrics = []
        for idx, col in enumerate(summary_df.columns):
            if col == "config_name":
                continue

            is_lower_better = True
            if col == "speed":
                is_lower_better = False

            metrics.append((idx, col, is_lower_better))

        for idx, metric, is_lower_better in metrics:
            vals = summary_df[metric].values
            arr = np.asarray(vals)

            if is_lower_better:
                best = arr.min()
            else:
                best = arr.max()

            for row_idx, val in enumerate(vals, 1):
                if val == best:
                    table[(row_idx, summary_df.columns.get_loc(metric))].get_text().set_fontweight("bold")

        fig.tight_layout()
        plt.savefig(summary_img_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        logger.info(f"Таблица с итоговыми метриками сохранена в {summary_img_path}")
