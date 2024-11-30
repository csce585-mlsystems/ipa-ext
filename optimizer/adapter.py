import concurrent.futures
import os
import re
import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..")))
from experiments.utils import logger
from experiments.utils.constants import LSTM_INPUT_SIZE, LSTM_PATH, NAMESPACE
from experiments.utils.pipeline_operations import (
    check_node_loaded,
    check_node_up,
    get_cpu_model_name,
    get_pod_name,
    is_terminating,
)
from experiments.utils.prometheus import PromClient
from optimizer import Optimizer, Pipeline
from optimizer.optimizer import Optimizer

prom_client = PromClient()

try:
    config.load_kube_config()
    kube_config = client.Configuration().get_default_copy()
except AttributeError:
    kube_config = client.Configuration()
    kube_config.assert_hostname = False
client.Configuration.set_default(kube_config)

kube_custom_api = client.CustomObjectsApi()


class Adapter:

    def __init__(
        self,
        pipeline_name: str,
        pipeline: Pipeline,
        node_names: List[str],
        adaptation_interval: int,
        optimization_method: Literal["gurobi", "brute-force", "q-learning"],
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
        from_storage: List[bool],
        baseline_mode: Optional[str] = None,
        central_queue: bool = False,
        debug_mode: bool = False,
        predictor_margin: int = 100,
        teleport_mode: bool = False,
        teleport_interval: int = 10,
        backup_predictor_type: str = "max",
        backup_predictor_duration: int = 2,
    ) -> None:
        """
        Args:
            pipeline_name (str): name of the pipeline
            pipeline (Pipeline): pipeline object
            adaptation_interval (int): adaptation interval of the pipeline
            optimization_method (Literal[gurobi, brute-force, q-learning])
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
        self.debug_mode = debug_mode
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
        self.monitoring = Monitoring(pipeline_name=self.pipeline_name, sla=self.pipeline.sla)
        self.predictor = Predictor(
            predictor_type=self.predictor_type,
            predictor_margin=predictor_margin,
            backup_predictor_type=self.backup_predictor_type,
            backup_predictor_duration=self.backup_predictor_duration,
        )
        self.central_queue = central_queue
        self.teleport_mode = teleport_mode
        self.teleport_interval = teleport_interval
        self.from_storage = {node_name: flag for node_name, flag in zip(node_names, from_storage)}

    def start_adaptation(self, workload=None):
        workload_timestep = 0 if workload else None
        time_interval = 0
        timestep = 0
        while True:
            check_interval = 5
            logger.info(
                f"Waiting for {check_interval} seconds before checking if the pipeline is up ...")
            for _ in tqdm.tqdm(range(check_interval)):
                time.sleep(1)
            pipeline_up = check_node_loaded(node_name="router")
            terminating = is_terminating(node_name="router")
            if pipeline_up and not terminating:
                logger.info("Found pipeline, starting adaptation ...")
                initial_config = self.extract_current_config()
                self.monitoring.get_router_pod_name()

                to_save_config = self.saving_config_builder(
                    to_apply_config=deepcopy(initial_config),
                    node_orders=deepcopy(self.node_names),
                    stage_wise_latencies=deepcopy(self.pipeline.stage_wise_latencies),
                    stage_wise_accuracies=deepcopy(self.pipeline.stage_wise_accuracies),
                    stage_wise_throughputs=deepcopy(self.pipeline.stage_wise_throughput),
                )
                self.monitoring.adaptation_step_report(
                    change_successful=[False] * len(self.node_names),
                    to_apply_config=to_save_config,
                    objective=None,
                    timestep=timestep,
                    monitored_load=[0],
                    time_interval=time_interval,
                    predicted_load=0,
                )
                break

        while True:
            logger.info("-" * 50)
            logger.info(f"Waiting {self.adaptation_interval} seconds to make next decision")
            logger.info("-" * 50)
            for _ in tqdm.tqdm(range(self.adaptation_interval)):
                time.sleep(1)
            if self.teleport_mode:
                workload_timestep += self.adaptation_interval
            pipeline_up = check_node_up(node_name="router")
            if not pipeline_up:
                logger.info("-" * 50)
                logger.info("No pipeline in the system, aborting adaptation process ...")
                logger.info("-" * 50)
                self.update_received_load(
                    workload[:workload_timestep] if self.teleport_mode else None)
                break

            time_interval += self.adaptation_interval
            timestep += 1
            rps_series = (workload[max(0, workload_timestep -
                                       self.monitoring_duration * 60):workload_timestep]
                          if self.teleport_mode else self.monitoring.rps_monitor(
                              monitoring_duration=self.monitoring_duration))
            if rps_series is None:
                continue
            predicted_load = self.predictor.predict(rps_series)
            logger.info("-" * 50)
            logger.info(f"\nPredicted Load: {predicted_load}\n")
            logger.info("-" * 50)
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
            if "objective" in optimal.columns:
                objective = optimal[[
                    "accuracy_objective", "resource_objective", "batch_objective", "objective"
                ]]
                new_configs = self.output_parser(optimal)
                logger.info("-" * 50)
                logger.info(f"Candidate configs:\n{new_configs}")
                logger.info("-" * 50)

                pipeline_up = check_node_up(node_name="router")
                if not pipeline_up:
                    logger.info("-" * 50)
                    logger.info("No pipeline in the system, aborting adaptation process ...")
                    logger.info("-" * 50)
                    self.update_received_load(
                        workload[:workload_timestep] if self.teleport_mode else None)
                    break

                to_apply_config = self.choose_config(new_configs)
                logger.info("-" * 50)
                logger.info(f"To be applied configs:\n{to_apply_config}")
                logger.info("-" * 50)

                config_change_results = (self.change_pipeline_config(to_apply_config)
                                         if to_apply_config else [])
            else:
                logger.info(
                    "Optimizer couldn't find any optimal solution; the pipeline will stay the same"
                )
                config_change_results = [False] * len(self.node_names)
                try:
                    to_apply_config = self.extract_current_config()
                except ApiException:
                    logger.info("-" * 50)
                    logger.info("No pipeline in the system, aborting adaptation process ...")
                    logger.info("-" * 50)
                    self.update_received_load()
                    break
                objective = None

            if to_apply_config:
                to_save_config = self.saving_config_builder(
                    to_apply_config=deepcopy(to_apply_config),
                    node_orders=deepcopy(self.node_names),
                    stage_wise_latencies=deepcopy(self.pipeline.stage_wise_latencies),
                    stage_wise_accuracies=deepcopy(self.pipeline.stage_wise_accuracies),
                    stage_wise_throughputs=deepcopy(self.pipeline.stage_wise_throughput),
                )
            self.monitoring.adaptation_step_report(
                to_apply_config=to_save_config if to_apply_config else None,
                objective=objective,
                timestep=timestep,
                time_interval=time_interval,
                monitored_load=rps_series,
                predicted_load=predicted_load,
                change_successful=config_change_results,
            )

    def output_parser(
            self, optimizer_output: pd.DataFrame) -> List[Dict[str, Dict[str, Union[str, int]]]]:
        return [{
            task_name: {
                "cpu": row[f"task_{i}_cpu"],
                "replicas": int(row[f"task_{i}_replicas"]),
                "batch": int(row[f"task_{i}_batch"]),
                "variant": row[f"task_{i}_variant"],
            }
            for i, task_name in enumerate(self.node_names)
        } for _, row in optimizer_output.iterrows()]

    def choose_config(
        self, new_configs: List[Dict[str, Dict[str, Union[str, int]]]]
    ) -> Optional[Dict[str, Dict[str, Union[str, int]]]]:
        try:
            current_config = self.extract_current_config()
        except ApiException:
            return None
        config_scores = [
            sum(-1 for node, cfg in new_config.items() for key, value in cfg.items()
                if (key == "variant" and value != current_config[node][key]) or (
                    key == "batch" and str(value) != current_config[node][key]))
            for new_config in new_configs
        ]
        if not config_scores:
            return None
        chosen_index = config_scores.index(max(config_scores))
        return new_configs[chosen_index]

    def extract_current_config(self) -> Dict[str, Dict[str, Union[str, int]]]:
        current_config = {}
        for node_name in self.node_names:
            try:
                raw_config = kube_custom_api.get_namespaced_custom_object(
                    group="machinelearning.seldon.io",
                    version="v1",
                    namespace=NAMESPACE,
                    plural="seldondeployments",
                    name=node_name,
                )
                component = raw_config["spec"]["predictors"][0]["componentSpecs"][0]
                env_vars = {
                    env["name"]: env["value"]
                    for env in component["spec"]["containers"][0].get("env", [])
                }
                node_config = {
                    "replicas":
                    component.get("replicas", 1),
                    "variant":
                    env_vars.get("MODEL_VARIANT", ""),
                    "cpu":
                    int(component["spec"]["containers"][0]["resources"]["requests"]["cpu"]),
                    "batch":
                    int(env_vars.get("MLSERVER_MODEL_MAX_BATCH_SIZE", 1))
                    if not self.central_queue else None,
                }
                if self.central_queue:
                    queue_config = kube_custom_api.get_namespaced_custom_object(
                        group="machinelearning.seldon.io",
                        version="v1",
                        namespace=NAMESPACE,
                        plural="seldondeployments",
                        name=f"queue-{node_name}",
                    )
                    queue_env = {
                        env["name"]: env["value"]
                        for env in queue_config["spec"]["predictors"][0]["componentSpecs"][0]
                        ["spec"]["containers"][0].get("env", [])
                    }
                    node_config["batch"] = int(queue_env.get("MLSERVER_MODEL_MAX_BATCH_SIZE", 1))
                current_config[node_name] = {k: v for k, v in node_config.items() if v is not None}
            except KeyError:
                logger.error(f"Configuration for node {node_name} is incomplete.")
                raise
        return current_config

    def change_pipeline_config(self, config: Dict[str, Dict[str, int]]) -> List[bool]:
        """Change the existing configuration based on the optimizer output.

        Args:
            config (Dict[str, Dict[str, int]]): Configuration dictionary for each node.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.change_node_config, config.items()))
        return results

    def change_node_config(self, inputs: Tuple[str, Dict[str, Any]]) -> bool:
        """
        Change the existing configuration based on the optimizer output.

        Args:
            inputs (Tuple[str, Dict[str, Any]]): A tuple containing the node name and
                its configuration.
        """
        node_name, node_config = inputs
        deployment_config = kube_custom_api.get_namespaced_custom_object(
            group="machinelearning.seldon.io",
            version="v1",
            namespace=NAMESPACE,
            plural="seldondeployments",
            name=node_name,
        )
        predictor_spec = deployment_config["spec"]["predictors"][0]["componentSpecs"][0]["spec"]
        containers = predictor_spec["containers"][0]

        # Update replicas and CPU resources
        deployment_config["spec"]["predictors"][0]["componentSpecs"][0]["replicas"] = node_config[
            "replicas"]
        containers["resources"]["limits"]["cpu"] = str(node_config["cpu"])
        containers["resources"]["requests"]["cpu"] = str(node_config["cpu"])

        # Update environment variables
        env_vars = containers.get("env", [])
        for env_var in env_vars:
            if env_var["name"] == "MODEL_VARIANT":
                env_var["value"] = node_config["variant"]
                if self.from_storage.get(node_name, False):
                    init_containers = predictor_spec.get("initContainers", [])
                    if init_containers and node_name not in ["yolo", "resnet-human"]:
                        init_args = init_containers[0].get("args", [])
                        init_containers[0]["args"] = [
                            re.sub(r"/([^/]+)$", f"/{node_config['variant']}", arg)
                            for arg in init_args
                        ]
            elif env_var["name"] == "MLSERVER_MODEL_MAX_BATCH_SIZE":
                env_var["value"] = "1"

        # Update queue configuration if central_queue is enabled
        if self.central_queue:
            queue_name = f"queue-{node_name}"
            queue_config = kube_custom_api.get_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1",
                namespace=NAMESPACE,
                plural="seldondeployments",
                name=queue_name,
            )
            queue_env_vars = queue_config["spec"]["predictors"][0]["componentSpecs"][0]["spec"][
                "containers"][0].get("env", [])
            for env_var in queue_env_vars:
                if env_var["name"] == "MLSERVER_MODEL_MAX_BATCH_SIZE":
                    env_var["value"] = str(node_config["batch"])

        # Retry mechanism for API calls
        number_of_retries = 3
        for attempt in range(number_of_retries):
            try:
                kube_custom_api.replace_namespaced_custom_object(
                    group="machinelearning.seldon.io",
                    version="v1",
                    namespace=NAMESPACE,
                    plural="seldondeployments",
                    name=node_name,
                    body=deployment_config,
                )
                if self.central_queue:
                    kube_custom_api.replace_namespaced_custom_object(
                        group="machinelearning.seldon.io",
                        version="v1",
                        namespace=NAMESPACE,
                        plural="seldondeployments",
                        name=queue_name,
                        body=queue_config,
                    )
                return True
            except ApiException as e:
                logger.warning(
                    f"Attempt {attempt + 1}: Failed to update {node_name} due to {e}. Retrying...")
                time.sleep(2)
        else:
            logger.error(f"Failed to update {node_name} after {number_of_retries} attempts.")
            return False

    def update_received_load(self, workload_of_teleport: Optional[List[float]] = None) -> None:
        """Extract the entire sent load during the experiment.

        Args:
            workload_of_teleport (Optional[List[float]]): Workload data if in teleport mode.
        """
        all_received_loads = (workload_of_teleport if workload_of_teleport is not None else
                              self.monitoring.rps_monitor(monitoring_duration=1000))
        self.monitoring.update_received_load(all_received_loads)

    def saving_config_builder(
        self,
        to_apply_config: Dict[str, Any],
        node_orders: List[str],
        stage_wise_latencies: List[float],
        stage_wise_accuracies: List[float],
        stage_wise_throughputs: List[float],
    ) -> Dict[str, Any]:
        saving_config = to_apply_config.copy()
        for node, latency, accuracy, throughput in zip(node_orders, stage_wise_latencies,
                                                       stage_wise_accuracies,
                                                       stage_wise_throughputs):
            saving_config[node].update({
                "latency": latency,
                "accuracy": accuracy,
                "throughput": throughput
            })
        return saving_config


class Monitoring:

    def __init__(self, pipeline_name: str, sla: float) -> None:
        self.pipeline_name = pipeline_name
        self.adaptation_report = {
            "timesteps": {},
            "metadata": {
                "sla": sla,
                "cpu_type": get_cpu_model_name()
            }
        }

    def rps_monitor(self, monitoring_duration: int = 1) -> List[int]:
        """
        Get the RPS of the router.

        Args:
            monitoring_duration (int): Duration in minutes.

        Returns:
            List[int]: RPS series.
        """
        rate = 2
        rps_series, _ = prom_client.get_input_rps(
            pod_name=self.router_pod_name,
            namespace="default",
            duration=monitoring_duration,
            container="router",
            rate=rate,
        )
        return rps_series

    def get_router_pod_name(self) -> None:
        self.router_pod_name = get_pod_name("router")[0]

    def adaptation_step_report(
        self,
        to_apply_config: Optional[Dict[str, Dict[str, Union[str, int]]]],
        objective: Optional[Dict[str, float]],
        timestep: int,
        time_interval: int,
        monitored_load: List[int],
        predicted_load: int,
        change_successful: List[bool],
    ) -> None:
        self.adaptation_report["change_successful"] = change_successful
        self.adaptation_report["timesteps"][timestep] = {
            "config": to_apply_config,
            "time_interval": time_interval,
            "monitored_load": monitored_load,
            "predicted_load": predicted_load,
            "accuracy_objective": float(objective["accuracy_objective"][0]) if objective else None,
            "resource_objective": float(objective["resource_objective"][0]) if objective else None,
            "batch_objective": float(objective["batch_objective"][0]) if objective else None,
            "objective": float(objective["objective"][0]) if objective else None,
        }

    def update_received_load(self, all_received_loads: List[float]) -> None:
        self.adaptation_report["metadata"]["received_load"] = all_received_loads


class Predictor:

    def __init__(
        self,
        predictor_type: str,
        backup_predictor_type: str = "reactive",
        backup_predictor_duration: int = 2,
        predictor_margin: int = 100,
    ) -> None:
        self.predictor_type = predictor_type
        self.backup_predictor = backup_predictor_type
        self.predictor_margin = predictor_margin
        self.backup_predictor_duration = backup_predictor_duration

        self.models = {
            "lstm": load_model(LSTM_PATH),
            "reactive": lambda l: l[-1],
            "max": max,
            "avg": lambda l: sum(l) / len(l) if l else 0,
            "arima": None,  # Defined dynamically in predict
        }
        self.model = self.models.get(predictor_type)
        self.backup_model = self.models.get(backup_predictor_type)

    def predict(self, series: List[int]) -> int:
        step = 10
        series_aggregated = [max(series[i:i + step]) for i in range(0, len(series), step)]

        sufficient_data = len(series_aggregated) >= (self.backup_predictor_duration * 60) // step
        if sufficient_data:
            if self.predictor_type == "lstm":
                model_input = tf.convert_to_tensor(
                    np.array(series_aggregated[-LSTM_INPUT_SIZE:]).reshape(-1, LSTM_INPUT_SIZE, 1),
                    dtype=tf.float32,
                )
                model_output = self.model.predict(model_input)[0][0]
            elif self.predictor_type == "arima":
                model = ARIMA(series_aggregated, order=(1, 0, 0)).fit()
                model_output = int(max(model.forecast(steps=2)))
            else:
                model_output = self.model(series_aggregated)
        else:
            model_output = self.backup_model(series_aggregated)

        # Apply a safety margin to the system
        predicted_load = round(model_output * (1 + self.predictor_margin / 100))
        return predicted_load
