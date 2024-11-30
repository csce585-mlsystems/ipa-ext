import os
import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model

# Get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.extend([
    os.path.normpath(os.path.join(project_dir, "..")),
    os.path.normpath(os.path.join(project_dir, "..", ".."))
])

from experiments.utils import logger
from experiments.utils.constants import LSTM_INPUT_SIZE, LSTM_PATH
from optimizer import Optimizer, Pipeline


class SimAdapter:

    def __init__(
        self,
        pipeline_name: str,
        pipeline: Pipeline,
        node_names: List[str],
        adaptation_interval: int,
        optimization_method: Literal["gurobi", "brute-force"],
        allocation_mode: Literal["base", "variable"],
        only_measured_profiles: bool,
        scaling_cap: int,
        batching_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        num_state_limit: int,
        monitoring_duration: int,
        predictor_type: str,
        baseline_mode: Optional[str] = None,
        backup_predictor_type: str = "max",
        backup_predictor_duration: int = 2,
        replica_factor: int = 1,
    ) -> None:
        """
        Args:
            pipeline_name (str): name of the pipeline
            pipeline (Pipeline): pipeline object
            adaptation_interval (int): adaptation interval of the pipeline
            optimization_method (Literal[gurobi, brute-force])
            allocation_mode (Literal[base;variable])
            only_measured_profiles (bool)
            scaling_cap (int)
            alpha (float): accuracy weight
            beta (float): resource weight
            gamma (float): batching weight
            num_state_limit (int): cap on the number of optimal states
            monitoring_duration (int): the monitoring
                daemon observing duration
        """
        self.pipeline_name = pipeline_name
        self.pipeline = pipeline
        self.node_names = node_names
        self.adaptation_interval = adaptation_interval
        self.backup_predictor_type = backup_predictor_type
        self.backup_predictor_duration = backup_predictor_duration
        self.optimizer = Optimizer(
            pipeline=pipeline,
            allocation_mode=allocation_mode,
            complete_profile=False,
            only_measured_profiles=only_measured_profiles,
            random_sample=False,
            baseline_mode=baseline_mode,
        )
        self.optimization_method = optimization_method
        self.scaling_cap = scaling_cap
        self.batching_cap = batching_cap
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_state_limit = num_state_limit
        self.monitoring_duration = monitoring_duration
        self.predictor_type = predictor_type
        self.monitoring = Monitoring(
            pipeline_name=self.pipeline_name,
            sla=self.pipeline.sla,
            base_allocations=self.optimizer.base_allocations(),
            stage_wise_slas=self.pipeline.stage_wise_slas,
        )
        self.predictor = Predictor(
            predictor_type=self.predictor_type,
            backup_predictor_type=self.backup_predictor_type,
            backup_predictor_duration=self.backup_predictor_duration,
        )
        self.replica_factor = replica_factor

    def start_adaptation(self, workload: List[int],
                         initial_config: Dict[str, Dict[str, Union[str, int]]]):
        # 0. Check if pipeline is up
        # 1. Use monitoring for periodically checking the status of
        #     the pipeline in terms of load
        # 2. Watches the incoming load in the system
        # 3. LSTM for predicting the load
        # 4. Get the existing pipeline state, batch size, model variant and replicas per
        #     each node
        # 5. Give the load and pipeline status to the optimizer
        # 6. Compare the optimal solutions from the optimizer
        #     to the existing pipeline's state
        # 7. Use the change config script to change the pipeline to the new config

        time_interval = 0
        old_config = deepcopy(initial_config)
        adaptation_steps = range(self.adaptation_interval, len(workload), self.adaptation_interval)
        for timestep in adaptation_steps:
            time_interval += self.adaptation_interval
            to_apply_config = None
            to_save_config = None
            objective = None

            rps_series = workload[max(0, timestep - self.monitoring_duration * 60):timestep]
            self.update_received_load(all_recieved_loads=rps_series)
            predicted_load = round(self.predictor.predict(rps_series))
            logger.info("-" * 50)
            logger.info(f"\nPredicted Load: {predicted_load}\n")
            logger.info("-" * 50)
            start_time = time.time()
            optimal = self.optimizer.optimize(
                optimization_method=self.optimization_method,
                scaling_cap=self.scaling_cap,
                batching_cap=self.batching_cap,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                arrival_rate=predicted_load,
                num_state_limit=self.num_state_limit,
            )
            duration = time.time() - start_time
            if "objective" in optimal.columns:
                objective = optimal[[
                    "accuracy_objective",
                    "resource_objective",
                    "batch_objective",
                    "objective",
                ]]
                new_configs = self.output_parser(optimal)
                to_apply_config = self.choose_config(new_configs, old_config)
            if to_apply_config is not None:
                to_save_config = self.saving_config_builder(
                    to_apply_config=deepcopy(to_apply_config),
                    node_orders=self.node_names.copy(),
                    stage_wise_latencies=self.pipeline.stage_wise_latencies.copy(),
                    stage_wise_accuracies=self.pipeline.stage_wise_accuracies.copy(),
                    stage_wise_throughputs=self.pipeline.stage_wise_throughput.copy(),
                )
            self.monitoring.adaptation_step_report(
                duration=duration,
                to_apply_config=to_save_config,
                objective=objective,
                timestep=str(timestep),
                time_interval=time_interval,
                monitored_load=rps_series,
                predicted_load=predicted_load,
            )
            old_config = deepcopy(to_apply_config)

    def output_parser(self, optimizer_output: pd.DataFrame):
        return [{
            task_name: {
                "cpu": row[f"task_{idx}_cpu"],
                "replicas": int(row[f"task_{idx}_replicas"]),
                "batch": int(row[f"task_{idx}_batch"]),
                "variant": row[f"task_{idx}_variant"],
            }
            for idx, task_name in enumerate(self.node_names)
        } for _, row in optimizer_output.iterrows()]

    def choose_config(self, new_configs: List[Dict[str, Dict[str, Union[str, int]]]],
                      current_config):
        # This should be from comparing with the
        # current config
        # easiest for now is to choose config with
        # with the least change from former config
        if current_config is None:
            return new_configs[0]
        config_scores = []
        for new_config in new_configs:
            score = 0
            for node_name, new_node_config in new_config.items():
                current_node_config = current_config.get(node_name, {})
                for knob, value in new_node_config.items():
                    current_value = current_node_config.get(knob)
                    if knob == "variant" and value != current_value:
                        score -= 1
                    if knob == "batch" and str(value) != str(current_value):
                        score -= 1
            config_scores.append(score)
        chosen_index = config_scores.index(max(config_scores))
        return new_configs[chosen_index]

    def update_received_load(self, all_recieved_loads: List[float]) -> None:
        """Extract the entire sent load during the experiment"""
        self.monitoring.update_received_load(all_recieved_loads)

    def saving_config_builder(
        self,
        to_apply_config: Dict[str, Any],
        node_orders: List[str],
        stage_wise_latencies: List[float],
        stage_wise_accuracies: List[float],
        stage_wise_throughputs: List[float],
    ):
        for idx, node in enumerate(node_orders):
            config = to_apply_config.get(node, {})
            config["latency"] = stage_wise_latencies[idx]
            config["accuracy"] = stage_wise_accuracies[idx]
            config["throughput"] = stage_wise_throughputs[idx]
            to_apply_config[node] = config
        return to_apply_config


class Monitoring:

    def __init__(
        self,
        pipeline_name: str,
        sla: float,
        base_allocations: Dict[str, Dict[str, int]],
        stage_wise_slas: Dict[str, float],
    ) -> None:
        self.pipeline_name = pipeline_name
        self.adaptation_report = {
            "timesteps": {},
            "metadata": {
                "sla": sla,
                "base_allocations": base_allocations,
                "stage_wise_slas": stage_wise_slas
            }
        }

    def adaptation_step_report(
        self,
        duration: float,
        to_apply_config: Dict[str, Dict[str, Union[str, int]]],
        objective: Optional[pd.DataFrame],
        timestep: str,
        time_interval: int,
        monitored_load: List[int],
        predicted_load: int,
    ):
        timestep_int = int(timestep)
        report = {
            "duration": duration,
            "config": to_apply_config,
            "time_interval": time_interval,
            "monitored_load": monitored_load,
            "predicted_load": predicted_load
        }
        if objective is not None:
            report.update({
                "accuracy_objective": float(objective["accuracy_objective"].iloc[0]),
                "resource_objective": float(objective["resource_objective"].iloc[0]),
                "batch_objective": float(objective["batch_objective"].iloc[0]),
                "objective": float(objective["objective"].iloc[0]),
            })
        else:
            report.update({
                "accuracy_objective": None,
                "resource_objective": None,
                "batch_objective": None,
                "objective": None,
            })
        self.adaptation_report["timesteps"][timestep_int] = report

    def update_received_load(self, all_recieved_loads: List[float]):
        self.adaptation_report["metadata"]["recieved_load"] = all_recieved_loads


class Predictor:

    def __init__(
        self,
        predictor_type: str,
        backup_predictor_type: str = "reactive",
        backup_predictor_duration: int = 2,
    ) -> None:
        self.predictor_type = predictor_type
        self.backup_predictor = backup_predictor_type
        predictors = {
            "lstm": load_model(LSTM_PATH),
            "reactive": lambda l: l[-1],
            "max": lambda l: max(l),
            "avg": lambda l: sum(l) / len(l),
            "arima": None,  # Defined in predict
        }
        self.model = predictors.get(predictor_type)
        self.backup_model = predictors.get(backup_predictor_type)
        self.backup_predictor_duration = backup_predictor_duration

    def predict(self, series: List[int]):
        step = 10  # Aggregate every 10 seconds
        series_aggregated = [max(series[i:i + step]) for i in range(0, len(series), step)]
        required_length = (self.backup_predictor_duration * 60) // step
        if len(series_aggregated) >= required_length:
            if self.predictor_type == "lstm":
                model_input = tf.convert_to_tensor(
                    series_aggregated[-LSTM_INPUT_SIZE:].reshape(-1, LSTM_INPUT_SIZE, 1),
                    dtype=tf.float32,
                )
                model_output = self.model.predict(model_input)[0][0]
            elif self.predictor_type == "arima":
                model = ARIMA(series_aggregated, order=(1, 0, 0))
                model_fit = model.fit()
                model_output = int(max(model_fit.forecast(steps=2)))
            else:
                model_output = self.model(series_aggregated)
        else:
            model_output = self.backup_model(series_aggregated)
        return model_output
