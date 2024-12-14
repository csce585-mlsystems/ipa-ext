import itertools
import os
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .models import Pipeline

# from functools import lru_cache


class Optimizer:
    __slots__ = (
        "pipeline",
        "allocation_mode",
        "complete_profile",
        "only_measured_profiles",
        "random_sample",
        "baseline_mode",
    )

    def __init__(
        self,
        pipeline: Pipeline,
        allocation_mode: str,
        complete_profile: bool,
        only_measured_profiles: bool,
        random_sample: bool,
        baseline_mode: Optional[str] = None,
    ) -> None:
        """
        Initialize the Optimizer.

        Args:
            pipeline (Pipeline): The pipeline object for optimization.
            allocation_mode (str): The allocation mode for CPU/GPU usage: fix|base|variable
                - fix: stays on the initial CPU allocation
                - base: finding the base allocation as explained in the paper
                - variable: search through CPU allocation as a configuration knob
            complete_profile (bool): Whether to log the complete result or not.
            only_measured_profiles (bool): If True, only profiles based on measured latency/throughput
                and does not use regression models.
            random_sample (bool): Whether to randomly sample states.
            baseline_mode (Optional[str]): Baseline approach mode: None|scale|switch|switch-scale.
        """
        self.pipeline = pipeline
        self.allocation_mode = allocation_mode
        self.complete_profile = complete_profile
        self.only_measured_profiles = only_measured_profiles
        self.random_sample = random_sample
        self.baseline_mode = baseline_mode

    # @lru_cache(maxsize=None)
    def accuracy_objective(self) -> float:
        """
        Returns the accuracy objective of the pipeline.

        Returns:
            float: The pipeline's accuracy objective.
        """
        return self.pipeline.pipeline_accuracy

    # @lru_cache(maxsize=None)
    def resource_objective(self) -> float:
        """
        Returns the resource objective of the pipeline.

        Returns:
            float: The pipeline's CPU usage as the resource objective.
        """
        return self.pipeline.cpu_usage

    # @lru_cache(maxsize=None)
    def batch_objective(self) -> float:
        """
        Computes the batch objective of the pipeline as the sum of all tasks' batch sizes.

        Returns:
            float: The cumulative batch size.
        """
        total_batch = 0
        for task in self.pipeline.inference_graph:
            total_batch += task.batch
        return total_batch

    # @lru_cache(maxsize=None)
    def objective(self, alpha: float, beta: float, gamma: float) -> Dict[str, float]:
        """
        Computes the combined objective function of the pipeline:
        objective = alpha*accuracy - beta*resource - gamma*batch

        Args:
            alpha (float): Weight for accuracy objective.
            beta (float): Weight for resource usage objective.
            gamma (float): Weight for batch size objective.

        Returns:
            Dict[str, float]: Dictionary containing individual and combined objectives.
        """
        acc_obj = alpha * self.accuracy_objective()
        res_obj = beta * self.resource_objective()
        bat_obj = gamma * self.batch_objective()
        return {
            "accuracy_objective": acc_obj,
            "resource_objective": res_obj,
            "batch_objective": bat_obj,
            "objective": acc_obj - res_obj - bat_obj,
        }

    # @lru_cache(maxsize=None)
    def constraints(self, arrival_rate: int) -> bool:
        """
        Checks if pipeline constraints (SLA and load sustainment) are met.

        Args:
            arrival_rate (int): The arrival rate to test.

        Returns:
            bool: True if SLA is met and load is sustainable, False otherwise.
        """
        return self.sla_is_met() and self.can_sustain_load(arrival_rate=arrival_rate)

    # @lru_cache(maxsize=None)
    def pipeline_latency_upper_bound(self, stage: str, variant_name: str) -> float:
        """
        Calculates the maximum latency of a given stage and variant by setting the largest batch.

        Args:
            stage (str): The pipeline stage name.
            variant_name (str): The variant name for that stage.

        Returns:
            float: The maximum model latency for the specified stage and variant.
        """
        inference_graph = deepcopy(self.pipeline.inference_graph)
        max_model = 0.0
        for task in inference_graph:
            if task.name == stage:
                task.model_switch(variant_name)
                task.change_batch(max(task.batches))
                max_model = task.model_latency
        return max_model

    # @lru_cache(maxsize=None)
    def latency_parameters(
        self, only_measured_profiles: bool
    ) -> Dict[str, Dict[str, Dict[int, float]]]:
        """
        Computes latency parameters for all stages, variants, and batch sizes.

        For each task variant and batch size, retrieves the measured latency.
        Missing batches are assigned a large dummy latency value.

        Args:
            only_measured_profiles (bool): If True, only measured profiles are used.

        Returns:
            Dict[str, Dict[str, Dict[int, float]]]: 
                [stage_name][variant_name][batch_size] = latency
        """
        model_latencies_parameters: Dict[str, Dict[str, Dict[int, float]]] = {}
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            stage_dict = {}
            for variant_name in task.variant_names:
                variant_dict = {}
                task.model_switch(variant_name)
                for batch_size in task.batches:
                    task.change_batch(batch_size)
                    variant_dict[batch_size] = task.model_latency
                stage_dict[variant_name] = variant_dict
            model_latencies_parameters[task.name] = stage_dict

        # Collect all distinct batch sizes
        all_batch_sizes = set()
        for stage_values in model_latencies_parameters.values():
            for variant_profile in stage_values.values():
                all_batch_sizes.update(variant_profile.keys())
        distinct_batches = sorted(all_batch_sizes)

        # Fill missing batches with dummy latency
        dummy_latency = 1000.0
        for stage, variants in model_latencies_parameters.items():
            for variant_name, variant_profile in variants.items():
                for batch in distinct_batches:
                    if batch not in variant_profile:
                        variant_profile[batch] = dummy_latency

        return model_latencies_parameters

    # @lru_cache(maxsize=None)
    def throughput_parameters(self) -> Tuple[List[int], Dict[str, Dict[str, Dict[int, float]]]]:
        """
        Computes throughput parameters for all stages, variants, and batch sizes.
        Missing batches are assigned a very small dummy throughput value.

        Returns:
            (List[int], Dict[str, Dict[str, Dict[int, float]]]):
                distinct_batches: A list of all distinct batch sizes.
                model_throughputs: [stage][variant][batch] = throughput
        """
        model_throughputs: Dict[str, Dict[str, Dict[int, float]]] = {}
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            stage_dict = {}
            for variant_name in task.variant_names:
                variant_dict = {}
                task.model_switch(variant_name)
                for batch_size in task.batches:
                    task.change_batch(batch_size)
                    variant_dict[batch_size] = task.throughput
                stage_dict[variant_name] = variant_dict
            model_throughputs[task.name] = stage_dict

        # Collect all distinct batch sizes
        all_batch_sizes = set()
        for stage_values in model_throughputs.values():
            for variant_profile in stage_values.values():
                all_batch_sizes.update(variant_profile.keys())
        distinct_batches = sorted(all_batch_sizes)

        # Fill missing batches with dummy throughput
        dummy_throughput = 0.00001
        for stage, variants in model_throughputs.items():
            for variant_name, variant_profile in variants.items():
                for batch in distinct_batches:
                    if batch not in variant_profile:
                        variant_profile[batch] = dummy_throughput

        return distinct_batches, model_throughputs

    # @lru_cache(maxsize=None)
    def accuracy_parameters(self) -> Dict[str, Dict[str, float]]:
        """
        Computes accuracy parameters for all stages and variants.

        Returns:
            Dict[str, Dict[str, float]]: [stage_name][variant_name] = accuracy
        """
        model_accuracies: Dict[str, Dict[str, float]] = {}
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            stage_dict = {}
            for variant_name in task.variant_names:
                task.model_switch(variant_name)
                stage_dict[variant_name] = task.accuracy
            model_accuracies[task.name] = stage_dict
        return model_accuracies

    # @lru_cache(maxsize=None)
    def base_allocations(self) -> Dict[str, Dict[str, float]]:
        """
        Computes base allocations of CPU or GPU for each stage and variant.

        Returns:
            Dict[str, Dict[str, float]]: [stage][variant] = base_allocation
        """
        base_allocs: Dict[str, Dict[str, float]] = {}
        inference_graph = deepcopy(self.pipeline.inference_graph)
        if self.pipeline.gpu_mode:
            for task in inference_graph:
                base_allocs[task.name] = {key: val.gpu for key, val in task.base_allocations.items()}
        else:
            for task in inference_graph:
                base_allocs[task.name] = {key: val.cpu for key, val in task.base_allocations.items()}
        return base_allocs

    def all_states(
        self,
        scaling_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        check_constraints: bool,
        arrival_rate: int,
        num_state_limit: int = None,
    ) -> pd.DataFrame:
        """
        Generates all possible states based on profiling data and configuration knobs.

        Args:
            scaling_cap (int): Max allowed scaling.
            alpha (float): Weight for accuracy objective.
            beta (float): Weight for resource objective.
            gamma (float): Weight for batch objective.
            check_constraints (bool): If True, only states that meet constraints are returned.
            arrival_rate (int): Arrival rate.
            num_state_limit (int, optional): Limit the number of states.

        Returns:
            pd.DataFrame: DataFrame of all possible states.
        """
        if num_state_limit is not None:
            state_counter = 0

        variant_names_list = []
        replicas_list = []
        batches_list = []
        allocations_list = []

        for task in self.pipeline.inference_graph:
            variant_names_list.append(task.variant_names)
            replicas_list.append(np.arange(1, scaling_cap + 1))
            batches_list.append(task.batches)
            if self.allocation_mode == "variable":
                if task.gpu_mode:
                    allocations_list.append(task.resource_allocations_gpu_mode)
                else:
                    allocations_list.append(task.resource_allocations_cpu_mode)
            elif self.allocation_mode == "fix":
                allocations_list.append([task.initial_allocation])
            elif self.allocation_mode == "base":
                pass
            else:
                raise ValueError(f"Invalid allocation_mode: {self.allocation_mode}")

        variant_names_product = list(itertools.product(*variant_names_list))
        replicas_product = list(itertools.product(*replicas_list))
        batches_product = list(itertools.product(*batches_list))

        if self.allocation_mode != "base":
            allocations_product = list(itertools.product(*allocations_list))
            all_combinations = itertools.product(
                variant_names_product, replicas_product, batches_product, allocations_product
            )
        else:
            all_combinations = itertools.product(
                variant_names_product, replicas_product, batches_product
            )

        if self.random_sample and num_state_limit is not None:
            all_combinations = random.sample(list(all_combinations), num_state_limit)

        states = []
        # Process states
        for combination in all_combinations:
            try:
                if self.allocation_mode != "base":
                    model_variant_tuple, replica_tuple, batch_tuple, allocation_tuple = combination
                else:
                    model_variant_tuple, replica_tuple, batch_tuple = combination
                    allocation_tuple = None

                # Apply configuration
                for task_id_i, task in enumerate(self.pipeline.inference_graph):
                    task.model_switch(active_variant=model_variant_tuple[task_id_i])
                    task.re_scale(replica=replica_tuple[task_id_i])
                    task.change_batch(batch=batch_tuple[task_id_i])
                    if self.allocation_mode != "base":
                        task.change_allocation(active_allocation=allocation_tuple[task_id_i])

                ok_to_add = not check_constraints or self.constraints(arrival_rate=arrival_rate)

                if ok_to_add:
                    state = {}
                    # record complete profile if needed
                    if self.complete_profile:
                        for task_id_j, task_obj in enumerate(self.pipeline.inference_graph):
                            state[f"task_{task_id_j}_latency"] = task_obj.latency
                            state[f"task_{task_id_j}_throughput"] = task_obj.throughput
                            state[f"task_{task_id_j}_throughput_all_replicas"] = task_obj.throughput_all_replicas
                            state[f"task_{task_id_j}_accuracy"] = task_obj.accuracy
                            state[f"task_{task_id_j}_measured"] = task_obj.measured
                            state[f"task_{task_id_j}_cpu_all_replicas"] = task_obj.cpu_all_replicas
                            state[f"task_{task_id_j}_gpu_all_replicas"] = task_obj.gpu_all_replicas

                        state["pipeline_accuracy"] = self.pipeline.pipeline_accuracy
                        state["pipeline_latency"] = self.pipeline.pipeline_latency
                        state["pipeline_throughput"] = self.pipeline.pipeline_throughput
                        state["pipeline_cpu"] = self.pipeline.pipeline_cpu
                        state["pipeline_gpu"] = self.pipeline.pipeline_gpu
                        state["alpha"] = alpha
                        state["beta"] = beta
                        state["gamma"] = gamma
                        state["accuracy_objective"] = self.accuracy_objective()
                        state["resource_objective"] = self.resource_objective()
                        state["batch_objective"] = self.batch_objective()

                    # record minimal profile
                    for task_id_j, task_obj in enumerate(self.pipeline.inference_graph):
                        state[f"task_{task_id_j}_variant"] = task_obj.active_variant
                        state[f"task_{task_id_j}_cpu"] = task_obj.cpu
                        state[f"task_{task_id_j}_gpu"] = task_obj.gpu
                        state[f"task_{task_id_j}_batch"] = task_obj.batch
                        state[f"task_{task_id_j}_replicas"] = task_obj.replicas

                    state.update(self.objective(alpha=alpha, beta=beta, gamma=gamma))
                    states.append(state)
                    # print(f"state {state_counter} added.")
                    if num_state_limit is not None:
                        state_counter += 1
                        if state_counter == num_state_limit:
                            break
            except StopIteration:
                pass

        return pd.DataFrame(states)

    def brute_force(
        self,
        scaling_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        arrival_rate: int,
        num_state_limit: int = None,
    ) -> pd.DataFrame:
        """
        Performs a brute-force search over all states and returns the optimal state(s).

        Args:
            scaling_cap (int): Max scaling.
            alpha (float): Accuracy weight.
            beta (float): Resource usage weight.
            gamma (float): Batch size weight.
            arrival_rate (int): Input arrival rate.
            num_state_limit (int, optional): Limit states.

        Returns:
            pd.DataFrame: DataFrame of optimal states.
        """
        states = self.all_states(
            check_constraints=True,
            scaling_cap=scaling_cap,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            arrival_rate=arrival_rate,
            num_state_limit=num_state_limit,
        )
        if states.empty:
            print("No feasible states found using brute-force optimization.")
            return states
        optimal = states[states["objective"] == states["objective"].max()]
        # print(f"state {len(optimal)} optimal state(s) found using brute-force optimization.")
        return optimal

    def gurobi_optimizer(
        self,
        scaling_cap: int,
        batching_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        arrival_rate: int,
        num_state_limit: int,
        dir_path: str = None,
    ) -> pd.DataFrame:
        """
        Uses Gurobi to solve the optimization problem.

        Args:
            scaling_cap (int): Max allowed horizontal scaling.
            batching_cap (int): Max allowed batch size.
            alpha (float): Accuracy weight.
            beta (float): Resource weight.
            gamma (float): Batch weight.
            arrival_rate (int): Input arrival rate.
            num_state_limit (int): Limit solutions to retrieve.
            dir_path (str, optional): Directory path to write LP file.

        Returns:
            pd.DataFrame: DataFrame of solutions.
        """
        self.only_measured_profiles = True
        sla = self.pipeline.sla
        stages = self.pipeline.stage_wise_task_names
        stages_variants = self.pipeline.stage_wise_available_variants

        base_allocations = self.base_allocations()
        accuracy_parameters = self.accuracy_parameters()
        distinct_batches, throughput_parameters = self.throughput_parameters()
        latency_parameters = self.latency_parameters(
            only_measured_profiles=self.only_measured_profiles
        )
        distinct_batches = [b for b in distinct_batches if b <= batching_cap]

        def func_q(batch: int, arrival_rate_: int) -> float:
            # queueing delay approximation
            if arrival_rate_ == 0:
                return 0.0
            return (batch - 1) / arrival_rate_

        model = gp.Model("pipeline")
        gurobi_variants = [(stg, var) for stg in stages for var in stages_variants[stg]]
        i = model.addVars(gurobi_variants, name="i", vtype=GRB.BINARY)
        n = model.addVars(stages, name="n", vtype=GRB.INTEGER, lb=1, ub=scaling_cap)
        b = model.addVars(stages, distinct_batches, name="b", vtype=GRB.BINARY)
        aux_batch = model.addVars(gurobi_variants, distinct_batches, name="aux", vtype=GRB.BINARY)

        # Constraints
        # Throughput constraints
        for stage in stages:
            for variant in stages_variants[stage]:
                for batch in distinct_batches:
                    model.addGenConstrAnd(
                        aux_batch[stage, variant, batch],
                        [i[stage, variant], b[stage, batch]],
                        name=f"andconstr-batch-variant-{stage}-{variant}-{batch}"
                    )
                    model.addConstr(
                        (aux_batch[stage, variant, batch] == 1) >>
                        (n[stage] * throughput_parameters[stage][variant][batch] >= arrival_rate)
                    )

        # Latency constraint
        model.addQConstr(
            gp.quicksum(
                latency_parameters[stage][variant][batch] * i[stage, variant] * b[stage, batch] +
                func_q(batch, arrival_rate)
                for stage in stages
                for variant in stages_variants[stage]
                for batch in distinct_batches
            ) <= sla,
            name="latency",
        )

        # One batch per stage
        for stage in stages:
            model.addConstr(
                gp.quicksum(b[stage, batch] for batch in distinct_batches) == 1,
                name=f"single-batch-{stage}"
            )

        # Baseline modes
        if self.baseline_mode == "scale":
            for task in self.pipeline.inference_graph:
                model.addConstr(i[task.name, task.active_variant] == 1, name=f"only-scale-task-{task.name}")
        elif self.baseline_mode == "switch":
            for task in self.pipeline.inference_graph:
                model.addConstr(n[task.name] == task.replicas, name=f"only-switch-task-{task.name}")

        # Only one active variant per stage
        for stage in stages:
            model.addConstr(
                gp.quicksum(i[stage, variant] for variant in stages_variants[stage]) == 1,
                name=f"one_model_{stage}"
            )

        # Accuracy objective
        if self.pipeline.accuracy_method == "multiply":
            if len(stages) <= 2:
                # two-stage multiply accuracy
                accuracy_objective = 1.0
                for stage in stages:
                    stage_accuracy_expr = gp.quicksum(
                        accuracy_parameters[stage][variant] * i[stage, variant]
                        for variant in stages_variants[stage]
                    )
                    # For multiply, we approximate by introducing a product of linear terms:
                    # With 2 stages max in the simplified scenario, we can handle directly:
                    # accuracy_objective *= stage_accuracy_expr won't directly work in Gurobi
                    # For a 2-stage scenario, just handle it as a separate variable and constraint:
                # If exactly 2 stages:
                if len(stages) == 2:
                    # We'll create variables for individual stage accuracies
                    accuracy_stage_0 = model.addVar(lb=0, ub=1, name="accuracy_stage_0")
                    accuracy_stage_1 = model.addVar(lb=0, ub=1, name="accuracy_stage_1")
                    model.addConstr(
                        accuracy_stage_0 == gp.quicksum(
                            accuracy_parameters[stages[0]][variant] * i[stages[0], variant]
                            for variant in stages_variants[stages[0]]
                        ),
                        name="accuracy_stage_0_def"
                    )
                    model.addConstr(
                        accuracy_stage_1 == gp.quicksum(
                            accuracy_parameters[stages[1]][variant] * i[stages[1], variant]
                            for variant in stages_variants[stages[1]]
                        ),
                        name="accuracy_stage_1_def"
                    )
                    accuracy_objective_var = model.addVar(lb=0, ub=1, name="accuracy_objective")
                    model.addGenConstrMul(accuracy_objective_var, accuracy_stage_0, accuracy_stage_1, name="accuracy_multiply")
                    accuracy_objective = accuracy_objective_var
                else:
                    # Handle multi-stage multiply accuracy
                    first_stage_variants = stages_variants[stages[0]]
                    second_stage_variants = stages_variants[stages[1]]
                    third_stage_variants = stages_variants[stages[2]]
                    all_pipeline_variant_combinations = list(
                        itertools.product(first_stage_variants, second_stage_variants, third_stage_variants)
                    )
                    all_comb_i = model.addVars(all_pipeline_variant_combinations, name="all_comb_i", vtype=GRB.INTEGER, lb=0, ub=1)
                    accuracy_objective_var = model.addVar(name="accuracy_objective", vtype=GRB.CONTINUOUS, lb=0, ub=1)
                    model.addConstr(
                        gp.quicksum(all_comb_i[combination] for combination in all_pipeline_variant_combinations) == 1,
                        name='one-model-combs'
                    )
                    for combination in all_pipeline_variant_combinations:
                        model.addConstr(
                            (all_comb_i[combination] == 1) >>
                            ((i[stages[0], combination[0]] +
                              i[stages[1], combination[1]] +
                              i[stages[2], combination[2]]) == 3),
                            name=f"accuracy_combo_{combination}"
                        )
                        combination_accuracy = (
                            accuracy_parameters[stages[0]][combination[0]] *
                            accuracy_parameters[stages[1]][combination[1]] *
                            accuracy_parameters[stages[2]][combination[2]]
                        )
                        model.addConstr(
                            (all_comb_i[combination] == 1) >> (accuracy_objective_var == combination_accuracy),
                            name=f"accuracy_value_{combination}"
                        )
                    accuracy_objective = accuracy_objective_var
        elif self.pipeline.accuracy_method == "sum":
            accuracy_objective = gp.quicksum(
                accuracy_parameters[stage][variant] * i[stage, variant]
                for stage in stages
                for variant in stages_variants[stage]
            )
        elif self.pipeline.accuracy_method == "average":
            accuracy_objective = gp.quicksum(
                accuracy_parameters[stage][variant] * i[stage, variant] * (1.0 / len(stages))
                for stage in stages
                for variant in stages_variants[stage]
            )
        else:
            raise ValueError(f"Invalid accuracy method {self.pipeline.accuracy_method}")

        resource_objective = gp.quicksum(
            base_allocations[stage][variant] * n[stage] * i[stage, variant]
            for stage in stages
            for variant in stages_variants[stage]
        )
        batch_objective = gp.quicksum(
            batch * b[stage, batch]
            for stage in stages
            for batch in distinct_batches
        )

        model.setObjective(
            alpha * accuracy_objective - beta * resource_objective - gamma * batch_objective,
            GRB.MAXIMIZE
        )

        # Gurobi parameters
        model.Params.PoolSearchMode = 2
        model.Params.LogToConsole = 0
        model.Params.PoolSolutions = num_state_limit
        model.Params.PoolGap = 0.0
        model.Params.NonConvex = 2

        model.optimize()

        if dir_path is not None:
            model.write(os.path.join(dir_path, "unmeasured.lp"))

        states = []
        for solution_count in range(model.SolCount):
            model.Params.SolutionNumber = solution_count
            all_vars = {v.varName: v.Xn for v in model.getVars()}

            i_var_output = {key: round(value) for key, value in all_vars.items() if key.startswith("i[")}
            n_var_output = {key: round(value) for key, value in all_vars.items() if key.startswith("n[")}
            b_var_output = {key: round(value) for key, value in all_vars.items() if key.startswith("b[")}

            i_output: Dict[str, str] = {}
            for stage in stages:
                chosen_variants = [(var, val) for (st, var), val in i_var_output.items() if st == stage and val == 1]
                if chosen_variants:
                    i_output[stage] = chosen_variants[0][0]

            n_output: Dict[str, int] = {}
            for stage in stages:
                vals = [val for key, val in n_var_output.items() if stage in key]
                n_output[stage] = vals[0] if vals else 1

            b_output: Dict[str, int] = {}
            for stage in stages:
                stage_batches = [(batch, val) for (st, batch), val in b_var_output.items() if st == stage and val == 1]
                if stage_batches:
                    b_output[stage] = stage_batches[0][0]

            # Apply solution to pipeline
            for task_id, stage in enumerate(stages):
                self.pipeline.inference_graph[task_id].model_switch(i_output[stage])
                self.pipeline.inference_graph[task_id].re_scale(n_output[stage])
                self.pipeline.inference_graph[task_id].change_batch(b_output[stage])

            state = {}
            if self.complete_profile:
                for task_id_j, task_obj in enumerate(self.pipeline.inference_graph):
                    state[f"task_{task_id_j}_latency"] = task_obj.latency
                    state[f"task_{task_id_j}_throughput"] = task_obj.throughput
                    state[f"task_{task_id_j}_throughput_all_replicas"] = task_obj.throughput_all_replicas
                    state[f"task_{task_id_j}_accuracy"] = task_obj.accuracy
                    state[f"task_{task_id_j}_measured"] = task_obj.measured
                    state[f"task_{task_id_j}_cpu_all_replicas"] = task_obj.cpu_all_replicas
                    state[f"task_{task_id_j}_gpu_all_replicas"] = task_obj.gpu_all_replicas

                state["pipeline_accuracy"] = self.pipeline.pipeline_accuracy
                state["pipeline_latency"] = self.pipeline.pipeline_latency
                state["pipeline_throughput"] = self.pipeline.pipeline_throughput
                state["pipeline_cpu"] = self.pipeline.pipeline_cpu
                state["pipeline_gpu"] = self.pipeline.pipeline_gpu
                state["alpha"] = alpha
                state["beta"] = beta
                state["gamma"] = gamma
                state["accuracy_objective"] = self.accuracy_objective()
                state["resource_objective"] = self.resource_objective()
                state["batch_objective"] = self.batch_objective()

            for task_id_j, task_obj in enumerate(self.pipeline.inference_graph):
                state[f"task_{task_id_j}_variant"] = task_obj.active_variant
                state[f"task_{task_id_j}_cpu"] = task_obj.cpu
                state[f"task_{task_id_j}_gpu"] = task_obj.gpu
                state[f"task_{task_id_j}_batch"] = task_obj.batch
                state[f"task_{task_id_j}_replicas"] = task_obj.replicas

            state.update(self.objective(alpha=alpha, beta=beta, gamma=gamma))
            states.append(state)

        if not states:
            print("No feasible solutions found using Gurobi optimization.")
            return pd.DataFrame([])

        print(f"Gurobi optimization recorded {len(states)} solution(s).")
        return pd.DataFrame(states)

    def optimize(
        self,
        optimization_method: str,
        scaling_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        arrival_rate: int,
        num_state_limit: int = None,
        batching_cap: int = None,
        dir_path: str = None,
    ) -> pd.DataFrame:
        """
        Main interface to run the optimization using either brute force, Gurobi, or Q-Learning.

        Args:
            optimization_method (str): "brute-force", "gurobi", or "q-learning".
            scaling_cap (int): Maximum horizontal scaling.
            alpha (float): Accuracy weight.
            beta (float): Resource weight.
            gamma (float): Batch weight.
            arrival_rate (int): Input arrival rate.
            num_state_limit (int, optional): Limit number of states/solutions.
            batching_cap (int, optional): Maximum batch size (for gurobi or q-learning).
            dir_path (str, optional): Directory to store model files.

        Returns:
            pd.DataFrame: Optimal states as a DataFrame.
        """
        if optimization_method == "brute-force":
            print("brute-force optimization selected.")
            optimal = self.brute_force(
                scaling_cap=scaling_cap,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit,
            )
        elif optimization_method == "gurobi":
            print("Gurobi optimization selected.")
            if batching_cap is None:
                raise ValueError("batching_cap must be provided for gurobi optimization.")
            optimal = self.gurobi_optimizer(
                scaling_cap=scaling_cap,
                batching_cap=batching_cap,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit,
                dir_path=dir_path,
            )
        elif optimization_method == "q-learning":
            print("Q-Learning optimization selected.")
            if batching_cap is None:
                raise ValueError("batching_cap must be provided for q-learning optimization.")
            optimal = self.q_learning(
                scaling_cap=scaling_cap,
                batching_cap=batching_cap,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit,
                dir_path=dir_path,
            )
        else:
            raise ValueError(f"Invalid optimization_method: {optimization_method}")
        return optimal

    def can_sustain_load(self, arrival_rate: int) -> bool:
        """
        Checks whether the current pipeline configuration can sustain the given load.

        Args:
            arrival_rate (int): The arrival rate to test.

        Returns:
            bool: True if pipeline can sustain the load, False otherwise.
        """
        for task in self.pipeline.inference_graph:
            if arrival_rate > task.throughput_all_replicas:
                return False
        return True

    def sla_is_met(self) -> bool:
        """
        Checks if the current pipeline SLA is met.

        Returns:
            bool: True if SLA is met, False otherwise.
        """
        return self.pipeline.pipeline_latency < self.pipeline.sla

    def find_load_bottlenecks(self, arrival_rate: int) -> List[int]:
        """
        Finds the bottleneck tasks that cannot handle the given arrival rate.

        Args:
            arrival_rate (int): The arrival rate to test.

        Returns:
            List[int]: List of task indices that are bottlenecks.
        """
        if self.can_sustain_load(arrival_rate=arrival_rate):
            raise ValueError("The load can be sustained! No bottleneck!")
        bottlenecks = []
        for task_id, task in enumerate(self.pipeline.inference_graph):
            if arrival_rate > task.throughput_all_replicas:
                bottlenecks.append(task_id)
        return bottlenecks

    def get_one_answer(self) -> Dict:
        """
        Returns one feasible answer from the optimizer.
        Currently not implemented.

        Returns:
            Dict: Empty dict as placeholder.
        """
        return {}

    def q_learning(
        self,
        scaling_cap: int,
        batching_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        arrival_rate: int,
        num_state_limit: int,
        dir_path: str = None,
    ) -> pd.DataFrame:
        """
        Q-Learning based optimization.

        We define:
        - State: For each stage, an index for variant, replicas, batch.
        - Actions: Change one aspect of a single stage (variant, replicas, batch).
        - Reward: Objective if feasible, else large negative.

        Steps:
        - Initialize Q-table as a dict: Q[(state),(action)] = float
        - Epsilon-greedy exploration
        - Run episodes to improve Q
        - Return best found state.

        Args:
            scaling_cap (int), batching_cap (int): define search space.
            alpha, beta, gamma (float): weights for objective.
            arrival_rate (int): load.
            num_state_limit (int): number of episodes (or steps).
            dir_path (Optional[str]): unused, for consistency.

        Returns:
            pd.DataFrame: Best found configuration.
        """

        stages = self.pipeline.stage_wise_task_names
        stages_variants = self.pipeline.stage_wise_available_variants
        distinct_batches, _ = self.throughput_parameters()
        distinct_batches = [b for b in distinct_batches if b <= batching_cap]

        # Map variants and batches to indices for easier handling
        stage_variant_options = [list(variants) for variants in stages_variants.values()]
        stage_batch_options = [distinct_batches for _ in stages]

        # State representation: tuple of ( (var_idx, replica, batch_idx), ... ) per stage
        def get_state():
            return tuple(
                (
                    stage_variant_options[s].index(self.pipeline.inference_graph[s].active_variant),
                    self.pipeline.inference_graph[s].replicas,
                    stage_batch_options[s].index(self.pipeline.inference_graph[s].batch),
                )
                for s in range(len(stages))
            )

        # Apply state to pipeline
        def apply_state(state):
            for s, (v_idx, r, b_idx) in enumerate(state):
                var = stage_variant_options[s][v_idx]
                batch_val = stage_batch_options[s][b_idx]
                self.pipeline.inference_graph[s].model_switch(var)
                self.pipeline.inference_graph[s].re_scale(r)
                self.pipeline.inference_graph[s].change_batch(batch_val)

        def compute_reward():
            # If constraints not met, large negative
            if not self.constraints(arrival_rate):
                return -1e6
            return self.objective(alpha=alpha, beta=beta, gamma=gamma)["objective"]

        # Actions: For each stage, we can:
        # 1. next variant, prev variant
        # 2. increase replicas, decrease replicas
        # 3. next batch, prev batch
        # This gives up to 6 actions per stage.
        # Total actions = 6 * num_stages
        # We'll encode action as (stage_index, action_type)
        # action_type in {0: next_var, 1: prev_var, 2: inc_replicas, 3: dec_replicas, 4: next_batch, 5: prev_batch}
        # We'll only execute valid actions (e.g. can't decrease replicas below 1).
        num_stages = len(stages)
        action_space = [(s, a_type) for s in range(num_stages) for a_type in range(6)]

        def next_var(v_idx, max_v):
            return (v_idx + 1) % max_v

        def prev_var(v_idx, max_v):
            return (v_idx - 1) % max_v

        def next_batch(b_idx, max_b):
            return (b_idx + 1) % max_b

        def prev_batch(b_idx, max_b):
            return (b_idx - 1) % max_b

        def get_next_state(state, action):
            s, a_type = action
            v_idx, r, b_idx = state[s]
            max_v = len(stage_variant_options[s])
            max_b = len(stage_batch_options[s])

            if a_type == 0:  # next variant
                v_idx = next_var(v_idx, max_v)
            elif a_type == 1:  # prev variant
                v_idx = prev_var(v_idx, max_v)
            elif a_type == 2:  # inc replicas
                if r < scaling_cap:
                    r += 1
            elif a_type == 3:  # dec replicas
                if r > 1:
                    r -= 1
            elif a_type == 4:  # next batch
                b_idx = next_batch(b_idx, max_b)
            elif a_type == 5:  # prev batch
                b_idx = prev_batch(b_idx, max_b)

            new_state = list(state)
            new_state[s] = (v_idx, r, b_idx)
            return tuple(new_state)

        # Q-learning parameters
        learning_rate = 0.1
        discount_factor = 0.95
        epsilon = 0.1
        episodes = num_state_limit if num_state_limit else 1000
        steps_per_episode = min(50, episodes)  # limit steps per episode

        Q = {}
        def get_Q(s, a):
            return Q.get((s, a), 0.0)

        best_state = None
        best_obj = -float('inf')

        for ep in range(episodes):
            # Random initial state: random configuration
            random_state = []
            for s in range(num_stages):
                v_idx = random.randint(0, len(stage_variant_options[s]) - 1)
                r = random.randint(1, scaling_cap)
                b_idx = random.randint(0, len(stage_batch_options[s]) - 1)
                random_state.append((v_idx, r, b_idx))
            state = tuple(random_state)
            apply_state(state)  # apply random init

            for step in range(steps_per_episode):
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.choice(action_space)
                else:
                    # choose best action according to Q
                    qs = [(get_Q(state, a), a) for a in action_space]
                    max_q = max(qs, key=lambda x: x[0])[0]
                    best_actions = [a for q, a in qs if q == max_q]
                    action = random.choice(best_actions) if best_actions else random.choice(action_space)

                new_state = get_next_state(state, action)
                apply_state(new_state)
                r = compute_reward()
                if r > best_obj and r != -1e6:
                    best_obj = r
                    best_state = new_state
                    # print(f"state {state_counter + 1} added.")
                
                # Update Q
                max_next = max(get_Q(new_state, a2) for a2 in action_space)
                old_q = get_Q(state, action)
                new_q = old_q + learning_rate * (r + discount_factor * max_next - old_q)
                Q[(state, action)] = new_q

                state = new_state

        if best_state is None:
            print("No feasible state found using Q-Learning optimization.")
            return pd.DataFrame([])

        # Apply best state and return result
        apply_state(best_state)
        obj_dict = self.objective(alpha=alpha, beta=beta, gamma=gamma)
        state_row = {}

        if self.complete_profile:
            for task_id_j, task_obj in enumerate(self.pipeline.inference_graph):
                state_row[f"task_{task_id_j}_latency"] = task_obj.latency
                state_row[f"task_{task_id_j}_throughput"] = task_obj.throughput
                state_row[f"task_{task_id_j}_throughput_all_replicas"] = task_obj.throughput_all_replicas
                state_row[f"task_{task_id_j}_accuracy"] = task_obj.accuracy
                state_row[f"task_{task_id_j}_measured"] = task_obj.measured
                state_row[f"task_{task_id_j}_cpu_all_replicas"] = task_obj.cpu_all_replicas
                state_row[f"task_{task_id_j}_gpu_all_replicas"] = task_obj.gpu_all_replicas

            state_row["pipeline_accuracy"] = self.pipeline.pipeline_accuracy
            state_row["pipeline_latency"] = self.pipeline.pipeline_latency
            state_row["pipeline_throughput"] = self.pipeline.pipeline_throughput
            state_row["pipeline_cpu"] = self.pipeline.pipeline_cpu
            state_row["pipeline_gpu"] = self.pipeline.pipeline_gpu
            state_row["alpha"] = alpha
            state_row["beta"] = beta
            state_row["gamma"] = gamma
            state_row["accuracy_objective"] = self.accuracy_objective()
            state_row["resource_objective"] = self.resource_objective()
            state_row["batch_objective"] = self.batch_objective()

        for task_id_j, task_obj in enumerate(self.pipeline.inference_graph):
            state_row[f"task_{task_id_j}_variant"] = task_obj.active_variant
            state_row[f"task_{task_id_j}_cpu"] = task_obj.cpu
            state_row[f"task_{task_id_j}_gpu"] = task_obj.gpu
            state_row[f"task_{task_id_j}_batch"] = task_obj.batch
            state_row[f"task_{task_id_j}_replicas"] = task_obj.replicas

        state_row.update(obj_dict)
        return pd.DataFrame([state_row])
