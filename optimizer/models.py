import math
from typing import Dict, List, Union

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class ResourceAllocation:

    def __init__(self, cpu: float = 0, gpu: float = 0) -> None:
        if cpu != 0 and gpu != 0:
            raise ValueError("For now only one of the CPU or GPU allocation is allowed")
        self.cpu = cpu
        self.gpu = gpu


class Profile:

    def __init__(
        self,
        batch: int,
        latency: float,
        measured: bool = True,
        measured_throughput=None,
    ) -> None:
        self.batch = batch
        self.latency = latency
        self.measured = measured
        if measured_throughput is not None:
            self.measured_throughput = measured_throughput

    @property
    def throughput(self):
        return self.measured_throughput if self.measured else (1 / self.latency) * self.batch

    def __eq__(self, other):
        if not isinstance(other, int):
            raise TypeError("batch size variables should be int")
        return self.batch == other


class Model:

    def __init__(
        self,
        name: str,
        resource_allocation: ResourceAllocation,
        measured_profiles: List[Profile],
        only_measured_profiles: bool,
        accuracy: float,
    ) -> None:
        self.name = name
        self.resource_allocation = resource_allocation
        self.measured_profiles = sorted(measured_profiles, key=lambda profile: profile.batch)
        self.only_measured_profiles = only_measured_profiles
        self.accuracy = accuracy / 100
        self.profiles, self.latency_model_params = self.regression_model()

    def regression_model(self) -> Union[List[Profile], Dict[str, float]]:
        train_x = np.array([profile.batch for profile in self.measured_profiles]).reshape(-1, 1)
        train_y = np.array([profile.latency for profile in self.measured_profiles]).reshape(-1, 1)
        all_x = train_x if self.only_measured_profiles else np.arange(
            self.min_batch, self.max_batch + 1).reshape(-1, 1)
        profiles = []

        if self.only_measured_profiles:
            profiles = [
                Profile(
                    batch=x,
                    latency=profile.latency,
                    measured=True,
                    measured_throughput=profile.measured_throughput,
                ) for x, profile in zip(all_x.flatten(), self.measured_profiles)
            ]
            model_parameters = {"coefficients": None, "intercept": None}
        else:
            poly = PolynomialFeatures(degree=2)
            train_x_poly = poly.fit_transform(train_x)
            test_x_poly = poly.transform(all_x)
            latency_model = LinearRegression().fit(train_x_poly, train_y)
            test_y = latency_model.predict(test_x_poly).flatten()

            linear_model = LinearRegression().fit(train_x, train_y)
            test_y_linear = linear_model.predict(all_x).flatten()
            test_y = np.where(test_y < 0, test_y_linear, test_y)

            profiles = sorted([
                Profile(batch=x, latency=y, measured=False)
                for x, y in zip(all_x.flatten(), test_y)
            ],
                              key=lambda profile: profile.batch)

            model_parameters = {
                "coefficients": latency_model.coef_[0],
                "intercept": latency_model.intercept_[0],
            }

            power_of_two_indices = [2**i - 1 for i in range(int(math.log2(len(profiles))) + 1)]
            profiles = [profiles[i] for i in power_of_two_indices if i < len(profiles)]

        return profiles, model_parameters

    @property
    def profiled_batches(self) -> List[int]:
        return [profile.batch for profile in self.measured_profiles]

    @property
    def min_batch(self) -> int:
        return min(self.profiled_batches)

    @property
    def max_batch(self) -> int:
        return max(self.profiled_batches)


class Task:

    def __init__(
        self,
        name: str,
        available_model_profiles: List[Model],
        active_variant: str,
        active_allocation: ResourceAllocation,
        replica: int,
        batch: int,
        allocation_mode: str,
        threshold: int,
        sla_factor: int,
        normalize_accuracy: bool,
        gpu_mode: bool = False,
    ) -> None:
        self.name = name
        self.available_model_profiles = available_model_profiles
        self.active_variant = active_variant
        self.active_allocation = active_allocation
        self.initial_allocation = active_allocation
        self.replicas = replica
        self.batch = batch
        self.allocation_mode = allocation_mode
        self.threshold = threshold
        self.sla_factor = sla_factor
        self.normalize_accuracy = normalize_accuracy
        self.gpu_mode = gpu_mode

        self.active_variant_index = self._find_active_variant_index()

    def _find_active_variant_index(self) -> int:
        for index, variant in enumerate(self.available_model_profiles):
            if variant.name != self.active_variant:
                continue
            allocation = variant.resource_allocation
            if (self.gpu_mode and self.active_allocation.gpu == allocation.gpu) or \
               (not self.gpu_mode and self.active_allocation.cpu == allocation.cpu):
                return index
        raise ValueError(
            f"no matching profile for the variant {self.active_variant} and allocation"
            f" of cpu: {self.active_allocation.cpu} and gpu: {self.active_allocation.gpu}")

    def remove_model_profiles_by_name(self, model_name: str):
        self.available_model_profiles = [
            profile for profile in self.available_model_profiles if profile.name != model_name
        ]

    def get_all_models_by_name(self, model_name: str) -> List[Model]:
        return [profile for profile in self.available_model_profiles if profile.name == model_name]

    def add_model_profile(self, model: Model):
        self.available_model_profiles.append(model)

    def add_model_profiles(self, models: List[Model]):
        self.available_model_profiles.extend(models)

    def model_switch(self, active_variant: str) -> None:
        for index, variant in enumerate(self.available_model_profiles):
            if variant.name != active_variant:
                continue
            allocation = variant.resource_allocation
            if (self.gpu_mode and self.active_allocation.gpu == allocation.gpu) or \
               (not self.gpu_mode and self.active_allocation.cpu == allocation.cpu):
                self.active_variant_index = index
                self.active_variant = active_variant
                if self.allocation_mode == "base":
                    self.set_to_base_allocation()
                return
        raise ValueError(
            f"no matching profile for the variant {active_variant} and allocation"
            f" of cpu: {self.active_allocation.cpu} and gpu: {self.active_allocation.gpu}")

    @property
    def num_variants(self) -> int:
        return len(self.variant_names)

    @property
    def sla(self) -> float:
        model_slas = {
            model: allocations[-1].profiles[0].latency * self.sla_factor
            for model, allocations in self._group_allocations().items()
        }
        task_sla = (sum(model_slas.values()) / len(model_slas)) * 5
        return task_sla

    def _group_allocations(self) -> Dict[str, List[Model]]:
        models = {key: [] for key in self.variant_names}
        for profile in self.available_model_profiles:
            models[profile.name].append(profile)
        return models

    @property
    def base_allocations(self) -> Dict[str, ResourceAllocation]:
        if self.allocation_mode != "base":
            return None
        models = self._group_allocations()
        base_allocation = {}
        allocation_num_sustains = {model: {} for model in models}

        for model_variant, allocations in models.items():
            for allocation in allocations:
                cpu = allocation.resource_allocation.cpu
                sustains = sum(profile.throughput >= self.threshold and profile.latency <= self.sla
                               for profile in allocation.profiles)
                allocation_num_sustains[model_variant][cpu] = sustains

        variant_orders = sorted(self.variants_accuracies.keys(),
                                key=lambda k: self.variants_accuracies[k])
        sample_allocation = sorted(allocation_num_sustains[variant_orders[0]].keys())
        for model in variant_orders:
            base_allocation[model] = None
            for cpu in sample_allocation:
                if allocation_num_sustains[model].get(cpu, 0) > 0:
                    base_allocation[model] = ResourceAllocation(cpu=cpu)
                    break
            if base_allocation[model] is None:
                raise ValueError(f"No responsive model profile to threshold {self.threshold}"
                                 f" or model sla {self.sla} was found for model variant {model}."
                                 " Consider changing the threshold or sla factor.")
        return base_allocation

    def set_to_base_allocation(self):
        self.change_allocation(active_allocation=self.base_allocations[self.active_variant])

    def change_allocation(self, active_allocation: ResourceAllocation) -> None:
        for index, variant in enumerate(self.available_model_profiles):
            if variant.name != self.active_variant:
                continue
            allocation = variant.resource_allocation
            if (self.gpu_mode and active_allocation.gpu == allocation.gpu) or \
               (not self.gpu_mode and active_allocation.cpu == allocation.cpu):
                self.active_variant_index = index
                self.active_allocation = active_allocation
                return
        raise ValueError(
            f"no matching profile for the variant {self.active_variant} and allocation"
            f" of cpu: {active_allocation.cpu} and gpu: {active_allocation.gpu}")

    def re_scale(self, replica: int) -> None:
        self.replicas = replica

    def change_batch(self, batch: int) -> None:
        self.batch = batch

    @property
    def variants_accuracies(self) -> Dict[str, float]:
        return dict(
            sorted({profile.name: profile.accuracy
                    for profile in self.available_model_profiles}.items(),
                   key=lambda item: item[1]))

    @property
    def variants_accuracies_normalized(self) -> Dict[str, float]:
        accuracies = sorted(self.variants_accuracies.values())
        variants = sorted(self.variants_accuracies.keys(),
                          key=lambda k: self.variants_accuracies[k])
        if len(accuracies) == 1:
            normalized = [1.0]
        else:
            normalized = list(np.linspace(0, 1, len(accuracies)))
        return {variant: norm for variant, norm in zip(variants, normalized)}

    @property
    def active_model(self) -> Model:
        return self.available_model_profiles[self.active_variant_index]

    @property
    def latency_model_params(self) -> Dict[str, float]:
        return self.active_model.latency_model_params

    @property
    def cpu(self) -> int:
        if self.gpu_mode:
            raise ValueError("The node is on GPU mode")
        return self.active_model.resource_allocation.cpu

    @property
    def gpu(self) -> float:
        return self.active_model.resource_allocation.gpu if self.gpu_mode else 0.0

    @property
    def cpu_all_replicas(self) -> int:
        if self.gpu_mode:
            raise ValueError("The node is on GPU mode")
        return self.active_model.resource_allocation.cpu * self.replicas

    @property
    def gpu_all_replicas(self) -> float:
        return self.active_model.resource_allocation.gpu * self.replicas if self.gpu_mode else 0.0

    @property
    def queue_latency(self) -> float:
        # Placeholder for queue latency computation
        return 0.0

    @property
    def model_latency(self) -> float:
        profile = next(profile for profile in self.active_model.profiles
                       if profile.batch == self.batch)
        return profile.latency

    @property
    def latency(self) -> float:
        return self.model_latency + self.queue_latency

    @property
    def throughput(self) -> float:
        profile = next(profile for profile in self.active_model.profiles
                       if profile.batch == self.batch)
        return profile.throughput

    @property
    def measured(self) -> bool:
        profile = next(profile for profile in self.active_model.profiles
                       if profile.batch == self.batch)
        return profile.measured

    @property
    def throughput_all_replicas(self) -> float:
        return self.throughput * self.replicas

    @property
    def accuracy(self) -> float:
        return (self.variants_accuracies_normalized.get(self.active_variant,
                                                        self.active_model.accuracy)
                if self.normalize_accuracy else self.active_model.accuracy)

    @property
    def variant_names(self) -> List[str]:
        return sorted({profile.name for profile in self.available_model_profiles})

    @property
    def batches(self) -> List[int]:
        return [profile.batch for profile in self.active_model.profiles]

    @property
    def resource_allocations_cpu_mode(self) -> List[ResourceAllocation]:
        unique_cpus = sorted(
            {profile.resource_allocation.cpu
             for profile in self.available_model_profiles})
        return [ResourceAllocation(cpu=cpu) for cpu in unique_cpus]

    @property
    def resource_allocations_gpu_mode(self) -> List[ResourceAllocation]:
        unique_gpus = sorted(
            {profile.resource_allocation.gpu
             for profile in self.available_model_profiles})
        return [ResourceAllocation(gpu=gpu) for gpu in unique_gpus]


class Pipeline:

    def __init__(
        self,
        inference_graph: List[Task],
        gpu_mode: bool,
        sla_factor: int,
        accuracy_method: str,
        normalize_accuracy: bool,
    ) -> None:
        self.inference_graph: List[Task] = inference_graph
        self.gpu_mode = gpu_mode
        self.sla_factor = sla_factor
        self.accuracy_method = accuracy_method
        self.normalize_accuracy = normalize_accuracy
        if not self.gpu_mode:
            gpu_tasks = [task.name for task in self.inference_graph if task.gpu_mode]
            if gpu_tasks:
                raise ValueError(
                    "pipeline is deployed on cpu, but the following tasks are on gpu: " +
                    ", ".join(gpu_tasks))

    def add_task(self, task: Task):
        self.inference_graph.append(task)

    def remove_task(self):
        if not self.inference_graph:
            raise IndexError("remove_task called on empty inference_graph")
        self.inference_graph.pop()

    @property
    def stage_wise_throughput(self) -> List[float]:
        return [task.throughput_all_replicas for task in self.inference_graph]

    @property
    def stage_wise_latencies(self) -> List[float]:
        return [task.latency for task in self.inference_graph]

    @property
    def sla(self) -> float:
        return sum(task.sla for task in self.inference_graph)

    @property
    def stage_wise_slas(self) -> Dict[str, float]:
        return {task.name: task.sla for task in self.inference_graph}

    @property
    def stage_wise_accuracies(self) -> List[float]:
        return [task.accuracy for task in self.inference_graph]

    @property
    def stage_wise_replicas(self) -> List[int]:
        return [task.replicas for task in self.inference_graph]

    @property
    def stage_wise_cpu(self) -> List[int]:
        return [task.cpu_all_replicas if not task.gpu_mode else 0 for task in self.inference_graph]

    @property
    def stage_wise_gpu(self) -> List[float]:
        return [task.gpu_all_replicas if task.gpu_mode else 0.0 for task in self.inference_graph]

    @property
    def stage_wise_task_names(self) -> List[str]:
        return [task.name for task in self.inference_graph]

    @property
    def stage_wise_available_variants(self) -> Dict[str, List[str]]:
        return {task.name: task.variant_names for task in self.inference_graph}

    @property
    def pipeline_cpu(self) -> int:
        return sum(self.stage_wise_cpu)

    @property
    def pipeline_gpu(self) -> float:
        return sum(self.stage_wise_gpu)

    @property
    def pipeline_latency(self) -> float:
        return sum(self.stage_wise_latencies)

    @property
    def pipeline_accuracy(self) -> float:
        tasks_accuracies = [
            task.variants_accuracies_normalized[task.active_variant]
            if self.normalize_accuracy else task.variants_accuracies[task.active_variant]
            for task in self.inference_graph
        ]

        if self.accuracy_method == "multiply":
            accuracy = 1.0
            for acc in tasks_accuracies:
                accuracy *= acc
        elif self.accuracy_method == "sum":
            accuracy = sum(tasks_accuracies)
        elif self.accuracy_method == "average":
            accuracy = sum(tasks_accuracies) / len(tasks_accuracies) if tasks_accuracies else 0.0
        else:
            raise ValueError(f"Unknown accuracy_method: {self.accuracy_method}")
        return accuracy

    @property
    def pipeline_throughput(self) -> float:
        if not self.stage_wise_throughput:
            return 0.0
        return min(self.stage_wise_throughput)

    @property
    def cpu_usage(self) -> int:
        return self.pipeline_cpu

    @property
    def gpu_usage(self) -> float:
        return self.pipeline_gpu

    @property
    def num_nodes(self) -> int:
        return len(self.inference_graph)

    def visualize(self):
        pass
