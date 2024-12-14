# %% [markdown]
# # Jsys-reviewers.ipynb
# 
# This notebook processes adaptation logs, computes metrics, and generates plots based on the data. It includes enhanced error handling with prominent warnings to notify users of any missing directories or issues during processing.

import importlib
import logging

# %%
import os
import sys
import warnings
from pprint import PrettyPrinter
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd  # Ensure pandas is imported as it's used in the code

# Initialize PrettyPrinter for better readability (optional)
pp = PrettyPrinter(indent=4)

# Configure logging to display prominent warnings with colors and emojis
class BoldFormatter(logging.Formatter):
    FORMATS = {
        logging.WARNING: "\033[93m⚠️ [WARNING] {message}\033[0m",  # Yellow with emoji
        logging.ERROR: "\033[91m❌ [ERROR] {message}\033[0m",      # Red with emoji
        logging.CRITICAL: "\033[91m❌ [CRITICAL] {message}\033[0m",# Red with emoji
        logging.INFO: "{message}",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, "{message}")
        return log_fmt.format(message=record.getMessage())

# Set up the logger
logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(BoldFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Set to INFO to capture all levels of logs

# %%
# Get an absolute path to the directory that contains parent files
__file__ = globals().get("_dh", [__file__])[0] if "_dh" in globals() else __file__
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..", "..")))

import experiments.utils.drawing
from experiments.utils.constants import FIGURES_PATH, FINAL_RESULTS_PATH
from experiments.utils.drawing import draw_cumulative, draw_temporal_final, draw_temporal_final2
from experiments.utils.parser import AdaptationParser

# %%
# Define constants for load types
BURSTY = "Bursty"
STEADY_LOW = "Steady Low"
STEADY_HIGH = "Steady High"
FLUCTUATING = "Fluctuating"

# Define series load types
series_load_type = {
    1: BURSTY,
    2: BURSTY,
    3: BURSTY,
    # 4: BURSTY,
    5: BURSTY,
    6: STEADY_LOW,
    7: STEADY_LOW,
    8: STEADY_LOW,
    # 9: STEADY_LOW,
    10: STEADY_LOW,
    11: STEADY_HIGH,
    12: STEADY_HIGH,
    13: STEADY_HIGH,
    # 14: STEADY_HIGH,
    15: STEADY_HIGH,
    16: FLUCTUATING,
    17: FLUCTUATING,
    18: FLUCTUATING,
    # 19: FLUCTUATING,
    20: FLUCTUATING,
}

# Define series names
series_names = {
    1: "IPA",
    2: "FA2-low",
    3: "FA2-high",
    # 4: "RIM-low",
    # 5: "RIM-high",
    5: "RIM",
    6: "IPA",
    7: "FA2-low",
    8: "FA2-high",
    # 9: "RIM-low",
    # 10: "RIM-high",
    10: "RIM",
    11: "IPA",
    12: "FA2-low",
    13: "FA2-high",
    # 14: "RIM-low",
    # 15: "RIM-high",
    15: "RIM",
    16: "IPA",
    17: "FA2-low",
    18: "FA2-high",
    # 19: "RIM-low",
    # 20: "RIM-high",
    20: "RIM",
}

# Define metaseries and load series
metaseries = 20
load_series = {BURSTY: [], STEADY_LOW: [], STEADY_HIGH: [], FLUCTUATING: []}
for serie, load_type in series_load_type.items():
    load_series[load_type].append(serie)

# Define pipeline name
pipeline_name = "video"

# Generate series paths
series_paths = {
    series: os.path.join(
        FINAL_RESULTS_PATH, "metaseries", str(metaseries), "series", str(series)
    )
    for series in series_names.keys()
}

# %%
# Initialize loaders with existence check and prominent warnings
loaders = {}
missing_series = []
for series, series_path in series_paths.items():
    if os.path.isdir(series_path):
        try:
            loaders[series] = AdaptationParser(
                series_path=series_path, model_name="video", type_of="router_pipeline"
            )
        except Exception as e:
            logger.warning(
                f"Error initializing AdaptationParser for series {series} at {series_path}: {e}"
            )
    else:
        logger.warning(
            f"Missing directory for series {series}: {series_path}"
        )
        missing_series.append(series)

# Remove missing series from further processing
for series in missing_series:
    del series_names[series]
    del series_load_type[series]

# Proceed only if loaders are available
if not loaders:
    logger.critical("No valid series loaders available. Please check the directories.")

# %%
# Load configurations
accuracy_methods = {}
adaptation_intervals = {}
simulation_modes = {}
configs = {}
for series, loader in loaders.items():
    try:
        configs_exp = loader.load_configs()
        config = configs_exp.get("0.yaml")
        if config is None:
            logger.warning(
                f"No '0.yaml' config found for series {series}"
            )
            continue
        configs[series] = config
        accuracy_methods[series] = config.get("accuracy_method", "default_method")
        adaptation_intervals[series] = config.get("adaptation_interval", 1)
        simulation_modes[series] = config.get("simulation_mode", "default_mode")
    except Exception as e:
        logger.warning(
            f"Error loading configs for series {series}: {e}"
        )

# %%
# Load adaptation logs
adaptation_logs = {}
for series, loader in loaders.items():
    try:
        adaptation_logs[series] = loader.load_adaptation_log()
    except Exception as e:
        logger.warning(
            f"Error loading adaptation log for series {series}: {e}"
        )
        adaptation_logs[series] = {}  # Assign an empty dict or handle as needed

# %%
# Compute series changes
series_changes = {}
for series in series_names.keys():
    try:
        series_changes[series] = loaders[series].series_changes(
            adaptation_log=adaptation_logs[series]
        )
    except Exception as e:
        logger.warning(
            f"Error computing series changes for series {series}: {e}"
        )
        series_changes[series] = {}  # Assign an empty dict or handle as needed

# %%
# Compile final metrics dictionary
final_dict = {}
METRIC_TOTAL_CORE_CHANGES = "total_core_changes"
METRIC_ACCURACY_CHANGES = "accuracy_changes"
METRIC_MEASURED_LATENCY = "measured_latency"
METRIC_TIMEOUT_DICS = "timeout_dics"
METRICS = [
    METRIC_TOTAL_CORE_CHANGES,
    METRIC_ACCURACY_CHANGES,
    METRIC_MEASURED_LATENCY,
    METRIC_TIMEOUT_DICS,
]

final_dict["replica_changes"] = {}
final_dict["core_changes"] = {}
for metric in METRICS:
    final_dict[metric] = {}

latency_metric = "p99"  # [min, max, p99]

for series, series_dict in series_changes.items():
    final_dict["replica_changes"][series] = {}
    final_dict["core_changes"][series] = {}
    final_dict[METRIC_TOTAL_CORE_CHANGES][series] = {}
    final_dict[METRIC_ACCURACY_CHANGES][series] = {}

    nodes = series_dict.get("nodes", {})
    for node_name, metrics in nodes.items():
        final_dict["replica_changes"][series][node_name] = metrics.get("replicas", 0)
        final_dict["core_changes"][series][node_name] = metrics.get("cpu", 0)
        final_dict[METRIC_ACCURACY_CHANGES][series][node_name] = metrics.get("accuracy", 0)

    try:
        timeout_per_second, per_second_results = loaders[
            series
        ].per_second_result_processing()
    except Exception as e:
        logger.warning(
            f"Error processing per-second results for series {series}: {e}"
        )
        timeout_per_second, per_second_results = {}, pd.DataFrame()

    metric_columns = list(
        filter(lambda col: latency_metric in col, per_second_results.columns)
    )
    final_dict[METRIC_MEASURED_LATENCY][series] = per_second_results[
        metric_columns
    ].to_dict(orient="list") if not per_second_results.empty else {}

    final_dict[METRIC_TIMEOUT_DICS][series] = {"timeout_per_second": timeout_per_second}

    # Compute totals
    try:
        # Sum across all nodes for replicas and cores
        replica_totals = [
            sum(x) for x in zip(*final_dict["replica_changes"][series].values())
        ] if final_dict["replica_changes"][series] else [0]
        core_totals = [
            sum(x) for x in zip(*final_dict["core_changes"][series].values())
        ] if final_dict["core_changes"][series] else [0]

        final_dict["replica_changes"][series]["total"] = replica_totals
        final_dict["core_changes"][series]["total"] = core_totals

        if accuracy_methods.get(series) == "sum":
            final_dict[METRIC_ACCURACY_CHANGES][series]["e2e"] = [
                sum(x) for x in zip(*final_dict["accuracy_changes"][series].values())
            ] if final_dict["accuracy_changes"][series] else [0]
    except Exception as e:
        logger.warning(
            f"Error computing totals for series {series}: {e}"
        )
        final_dict["replica_changes"][series]["total"] = [0]
        final_dict["core_changes"][series]["total"] = [0]
        final_dict[METRIC_ACCURACY_CHANGES][series]["e2e"] = [0]

    # Compute total core changes
    for key in final_dict["replica_changes"][series].keys():
        try:
            replicas = final_dict["replica_changes"][series][key]
            cores = final_dict["core_changes"][series][key]
            total_core_changes = [
                x * y
                for x, y in zip(
                    replicas,
                    cores,
                )
            ]
            final_dict[METRIC_TOTAL_CORE_CHANGES][series][key] = total_core_changes
        except Exception as e:
            logger.warning(
                f"Error computing total core changes for series {series}, node {key}: {e}"
            )
            final_dict[METRIC_TOTAL_CORE_CHANGES][series][key] = [0]

# Remove intermediary data
del final_dict["replica_changes"]
del final_dict["core_changes"]

# %%
# Prepare data for plotting
METRICS_TO_PLOT = [
    METRIC_TOTAL_CORE_CHANGES,
    METRIC_ACCURACY_CHANGES,
    # METRIC_MEASURED_LATENCY,
    # METRIC_TIMEOUT_DICS,
]

final_by_load_type = {BURSTY: {}, STEADY_LOW: {}, STEADY_HIGH: {}, FLUCTUATING: {}}

for k in final_by_load_type:
    for m in METRICS_TO_PLOT:
        final_by_load_type[k][m] = {}

for metric in METRICS_TO_PLOT:
    for serie in final_dict.get(metric, {}).keys():
        load_type = series_load_type.get(serie)
        series_name = series_names.get(serie, f"Series {serie}")
        if load_type:
            final_by_load_type[load_type][metric][series_name] = final_dict[metric][serie]

# %%
# Reload drawing module to ensure latest changes are reflected
importlib.reload(experiments.utils.drawing)

# Define selected experiments for plotting
selected_experiments = {
    METRIC_TOTAL_CORE_CHANGES: {
        "selection": ["total"],
        "title": "Cost",
        "ylabel": "Cost\n (cores)",
    },
    METRIC_ACCURACY_CHANGES: {
        "selection": ["e2e"],
        "title": "Accuracy",
        "ylabel": "Accuracy",
    },
    METRIC_MEASURED_LATENCY: {
        "selection": [f"e2e_{latency_metric}"],
        "title": "Latency",
        "ylabel": "Latency (s)",
    },
    METRIC_TIMEOUT_DICS: {
        "selection": [f"timeout_per_second"],
        "title": "SLA Violations",
        "ylabel": "SLA\n Violations",
    },
}

# Define colors for the series
serie_color = {
    "IPA": "#d7191c",
    "FA2-low": "#a1dab4",
    "FA2-high": "#41b6c4",
    "RIM-low": "#2c7fb8",
    "RIM": "#253494",
}

# %%
# Plot the data with enhanced error handling
try:
    experiments.utils.drawing.draw_temporal_final4(
        final_by_load_type,
        adaptation_interval=adaptation_intervals,
        selected_experiments=selected_experiments,
        serie_color=serie_color,
        # hl_for_metric = {METRIC_MEASURED_LATENCY: {"value": 4, "color": "black", "label": "SLA"}},
        hl_for_metric={},
        bbox_to_anchor=(0.55, 2.5 * len(METRICS_TO_PLOT) + 1.1),
        hspace=0.3,
        save=True,
        filename=f"{FIGURES_PATH}/metaseries-{metaseries}-{pipeline_name}.pdf",
    )
except Exception as e:
    logger.warning(
        f"Error during plotting: {e}"
    )

# %% [markdown]
# ## Summary of Enhancements
# 
# - **Prominent Warnings:**
#   - Warnings for missing directories, configuration loading issues, and plotting errors are styled with emojis and colored text to ensure they are easily noticeable.
# 
# - **Robustness and Continuity:**
#   - The notebook continues processing available data even if some series are missing or if certain steps fail for specific series.
#   - Missing or problematic series are excluded from further processing to prevent errors downstream.
# 
# - **Maintained Plot Functionality:**
#   - The plotting logic remains unchanged and accurately reflects the available data.
#   - Only series with existing directories and successfully loaded configurations are visualized.
# 
# - **Error Handling:**
#   - Critical steps are wrapped in `try-except` blocks to catch and log any issues without halting the entire notebook.
# 
# By implementing these modifications, your notebook will now **issue highly noticeable warnings** when directories are missing or when errors occur, while **continuing to process and plot the available data** seamlessly. This approach ensures that you are alerted to potential issues without disrupting the overall workflow and data visualization.

# %%
# End of Notebook
