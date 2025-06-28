import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("whisper-benchmark")


class ResultsAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.run_dir = self._create_run_directory()

    def _create_run_directory(self) -> Path:
        os.makedirs(self.output_dir, exist_ok=True)
        run_time = datetime.now().strftime("%d.%m.%Y_%H-%M")
        run_dir = self.output_dir / run_time
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Создана директория для результатов запуска: {run_dir}")
        return run_dir

    def save_results(
        self,
        results: Dict[str, Any],
        model: str,
        compute: str,
        config_name: Optional[str] = None,
    ) -> Path:
        if config_name:
            base = f"r_{config_name}"
        else:
            base = f"r_{model}_{compute}"

        base = base.replace("/", "_").replace("@", "_")

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
        avg_rows = []
        file_rows = []

        for config_name, config_results in results.items():
            config = config_results["config"]

            metrics = {}

            for file_name, file_result in config_results["files"].items():
                for metric_name, metric_value in file_result["metrics"].items():
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(metric_value)

                file_row = {
                    "config_name": config_name,
                    "file_name": file_name,
                    **config,
                    **file_result["metrics"],
                }

                file_rows.append(file_row)

            avg_row: Dict[str, Any] = {
                "config_name": config_name,
                **config,
            }

            for metric_name, metric_values in metrics.items():
                avg_row[metric_name] = np.mean(metric_values)

            avg_rows.append(avg_row)

        avg_df = pd.DataFrame(avg_rows)
        file_df = pd.DataFrame(file_rows)

        configs = [r["config"] for r in results.values()]
        base = "__".join([f"{c['config_name'] or 'unknown'}" for c in configs])
        csv_path = self._get_unique_path(self.run_dir, f"r_{base}", "csv")
        avg_df.to_csv(csv_path, index=False)
        logger.info(f"CSV с результатами сохранен в {csv_path}")
        self._create_visualizations(avg_df, file_df)

    def _make_histogram(
        self, df, metric, title, ylabel, filename, is_lower_better=True
    ):
        plt.figure(figsize=(19.2, 10.8))
        plt.bar(df["config_name"], df[metric])
        plt.title(title)
        plt.xlabel("Конфигурация")
        plt.ylabel(
            f"{ylabel} ({'меньше - лучше' if is_lower_better else 'больше - лучше'})"
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots_dir = self.run_dir / "plots"
        fp = self._get_unique_path(plots_dir, filename, "png")
        plt.savefig(fp)
        plt.close()

    def _make_plot(
        self, df, x, y, title, xlabel, ylabel, filename, is_lower_better=True
    ):
        plt.figure(figsize=(19.2, 10.8))
        for config_name in df["config_name"].unique():
            config_df = df[df["config_name"] == config_name].copy()
            if not config_df.empty:
                config_df = config_df.sort_values(by=x)
                plt.plot(config_df[x], config_df[y], linestyle="-", label=config_name)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(
            f"{ylabel} ({'меньше - лучше' if is_lower_better else 'больше - лучше'})"
        )
        if df["config_name"].nunique() > 1:
            plt.legend()
        plt.xticks(rotation=90 if x == "file_name" else 45)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plots_dir = self.run_dir / "plots"
        fp = self._get_unique_path(plots_dir, filename, "png")
        plt.savefig(fp)
        plt.close()

    def _create_visualizations(
        self, avg_df: pd.DataFrame, file_df: pd.DataFrame
    ) -> None:
        plots_dir = self.run_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)

        self._make_histogram(
            avg_df, "wer", "Word Error Rate (WER) по конфигурациям", "WER", "wer", True
        )

        time_speed_cols = [col for col in avg_df.columns if col.endswith("_speed")]
        for col in time_speed_cols:
            if col in avg_df.columns:
                stage_name = col.replace("_speed", "")
                self._make_histogram(
                    avg_df,
                    col,
                    f"Скорость этапа {stage_name} (длительность/время_этапа)",
                    f"{stage_name} speed",
                    f"{stage_name}_speed",
                    False,
                )

        # Графики для скоростей отдельных этапов в зависимости от длительности
        for col in time_speed_cols:
            if col in file_df.columns and file_df[col].notna().any():
                stage_name = col.replace("_speed", "")
                self._make_plot(
                    file_df,
                    "duration",
                    col,
                    f"Зависимость скорости этапа {stage_name} от длительности аудио",
                    "Длительность аудио (секунды)",
                    f"Скорость {stage_name}",
                    f"duration_vs_{stage_name}_speed",
                    False,
                )

        self._make_process_time_breakdown(avg_df, plots_dir)

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

        file_df["accuracy"] = 1 - file_df["wer"]
        self._make_plot(
            file_df,
            "file_name",
            "accuracy",
            "Точность обработки (1 - WER) по файлам",
            "Имя файла",
            "Точность",
            "accuracy_by_file",
            False,
        )
        logger.info(f"Графики сохранены в директории {plots_dir}")

        self._create_summary(avg_df, plots_dir)

    def _make_process_time_breakdown(self, avg_df, plots_dir):
        """Создает гистограмму с разбивкой времени на этапы в процентах"""
        time_cols = [
            col
            for col in avg_df.columns
            if col.endswith("_time") and col not in ["total_processing_time"]
        ]
        if not time_cols:
            return

        plt.figure(figsize=(14, 8))

        config_names = avg_df["config_name"].tolist()
        n_configs = len(config_names)

        # Собираем данные для разбивки по этапам
        stage_data = {}
        total_times = {}

        # Общее время для каждой конфигурации
        for i, config in enumerate(config_names):
            config_df = avg_df[avg_df["config_name"] == config]
            total_time = 0

            for col in time_cols:
                if col in config_df.columns:
                    time_val = config_df[col].values[0]
                    total_time += time_val

                    stage_name = col.replace("_time", "")
                    if stage_name not in stage_data:
                        stage_data[stage_name] = [0] * n_configs

                    stage_data[stage_name][i] = time_val

            total_times[config] = total_time

        # Преобразуем в проценты от общего времени
        for stage in stage_data:
            for i, config in enumerate(config_names):
                if total_times[config] > 0:
                    stage_data[stage][i] = (
                        stage_data[stage][i] / total_times[config]
                    ) * 100
                else:
                    stage_data[stage][i] = 0

        # Строим гистограмму
        bottom = np.zeros(n_configs)
        # Используем фиксированный набор цветов
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        for i, (stage, percentages) in enumerate(stage_data.items()):
            color_idx = i % len(colors)
            plt.bar(
                config_names,
                percentages,
                bottom=bottom,
                label=stage,
                color=colors[color_idx],
            )
            # Добавляем подписи значений
            for j, val in enumerate(percentages):
                if val > 0:
                    plt.text(
                        j,
                        bottom[j] + val / 2,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white",
                    )
            bottom += percentages

        plt.title("Разбивка времени обработки по этапам")
        plt.xlabel("Конфигурация")
        plt.ylabel("Доля времени, %")
        plt.xticks(rotation=45)
        plt.legend(title="Этапы обработки")
        plt.tight_layout()

        plot_path = plots_dir / "time_breakdown.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"График разбивки времени сохранен в {plot_path}")

    def _create_summary(self, avg_df, plots_dir):
        summary_cols = ["config_name", "wer", "avg_utilization", "max_memory_used_mb"]

        speed_cols = [col for col in avg_df.columns if col.endswith("_speed")]
        summary_cols.extend(speed_cols)

        summary_df = avg_df[summary_cols].copy()

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
        max_config_len = max(
            [len(str(x)) for x in summary_df["config_name"]] + [len("config_name")]
        )
        table.auto_set_column_width(col=list(range(len(summary_df.columns))))
        cell = table.get_celld()[(0, 0)]
        cell.set_width(0.08 + 0.01 * max(0, max_config_len - 10))

        metrics = []
        for idx, col in enumerate(summary_df.columns):
            if col == "config_name":
                continue

            is_lower_better = "speed" not in col
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
                    table[
                        (row_idx, summary_df.columns.get_loc(metric))
                    ].get_text().set_fontweight("bold")

        fig.tight_layout()
        plt.savefig(summary_img_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        logger.info(f"Таблица с итоговыми метриками сохранена в {summary_img_path}")
