import itertools
import math
import os
import random
from copy import deepcopy
from typing import Dict, List, Optional, Union

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .models import Pipeline


class Optimizer:

    def __init__(
        self,
        pipeline: Pipeline,
        allocation_mode: str,
        complete_profile: bool,
        only_measured_profiles: bool,
        random_sample: bool,
        baseline_mode: Optional[str] = None,
    ) -> None:
        """Initialize the Optimizer with pipeline and configuration parameters.

        Args:
            pipeline (Pipeline): Pipeline object for optimization.
            allocation_mode (str): Allocation mode for CPU usage ('fix', 'base', 'variable').
            complete_profile (bool): Whether to log the complete result.
            only_measured_profiles (bool): Use only measured profiles, not regression models.
            random_sample (bool): Whether to use random sampling for state generation.
            baseline_mode (Optional[str], optional): Baseline mode. Defaults to None.
        """
        self.pipeline = pipeline
        self.allocation_mode = allocation_mode
        self.complete_profile = complete_profile
        self.only_measured_profiles = only_measured_profiles
        self.random_sample = random_sample
        self.baseline_mode = baseline_mode

    def accuracy_objective(self) -> float:
        """Calculate the accuracy objective of the pipeline."""
        return self.pipeline.pipeline_accuracy

    def resource_objective(self) -> float:
        """Calculate the resource usage objective of the pipeline."""
        return self.pipeline.cpu_usage

    def batch_objective(self) -> float:
        """Calculate the batch objective of the pipeline."""
        return max(task.batch for task in self.pipeline.inference_graph)

    def objective(self, alpha: float, beta: float, gamma: float) -> Dict[str, float]:
        """Compute the weighted objectives of the pipeline."""
        accuracy = alpha * self.accuracy_objective()
        resource = beta * self.resource_objective()
        batch = gamma * self.batch_objective()
        return {
            "accuracy_objective": accuracy,
            "resource_objective": resource,
            "batch_objective": batch,
            "objective": accuracy - resource - batch
        }

    def constraints(self, arrival_rate: int) -> bool:
        """Check if the pipeline meets the SLA and can sustain the load."""
        return self.sla_is_met() and self.can_sustain_load(arrival_rate=arrival_rate)

    def pipeline_latency_upper_bound(self, stage: str, variant_name: str) -> float:
        """Calculate the maximum latency for a given stage and variant."""
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            if task.name == stage:
                task.model_switch(variant_name)
                task.change_batch(max(task.batches))
                return task.model_latency
        return 0.0

    def latency_parameters(
        self, only_measured_profiles: bool
    ) -> Union[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, Dict[str, float]]]]:
        """Generate latency parameters for all models and batches."""
        model_latencies = {
            task.name: {
                variant: {
                    batch: (task.change_batch(batch) or task.model_latency)
                    for batch in task.batches
                }
                for variant in task.variant_names
            }
            for task in deepcopy(self.pipeline.inference_graph)
        }

        distinct_batches = sorted({
            batch
            for variants in model_latencies.values()
            for variant in variants.values()
            for batch in variant.keys()
        })

        dummy_latency = 1000.0
        for variants in model_latencies.values():
            for variant in variants.values():
                for batch in distinct_batches:
                    variant.setdefault(batch, dummy_latency)

        return model_latencies

    def throughput_parameters(self) -> Dict[str, Dict[str, List[float]]]:
        """Generate throughput parameters for all models and batches."""
        model_throughputs = {
            task.name: {
                variant: {
                    batch: (task.change_batch(batch) or task.throughput)
                    for batch in task.batches
                }
                for variant in task.variant_names
            }
            for task in deepcopy(self.pipeline.inference_graph)
        }

        distinct_batches = sorted({
            batch
            for variants in model_throughputs.values()
            for variant in variants.values()
            for batch in variant.keys()
        })

        dummy_throughput = 0.00001
        for variants in model_throughputs.values():
            for variant in variants.values():
                for batch in distinct_batches:
                    variant.setdefault(batch, dummy_throughput)

        return model_throughputs

    def accuracy_parameters(self) -> Dict[str, Dict[str, float]]:
        """Generate accuracy parameters for all models."""
        return {
            task.name: {
                variant: (task.model_switch(variant) or task.accuracy)
                for variant in task.variant_names
            }
            for task in deepcopy(self.pipeline.inference_graph)
        }

    def base_allocations(self) -> Dict[str, Dict[str, float]]:
        """Retrieve base allocations for all models based on GPU mode."""
        return {
            task.name: {
                key: value.gpu if self.pipeline.gpu_mode else value.cpu
                for key, value in task.base_allocations.items()
            }
            for task in deepcopy(self.pipeline.inference_graph)
        }

    def q_learning_optimizer(
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
        self.only_measured_profiles = True  # Handle both cases using pre-calculated profiles
        sla = self.pipeline.sla
        stages = self.pipeline.stage_wise_task_names
        stages_variants = self.pipeline.stage_wise_available_variants

        # Retrieve parameters
        base_allocations = self.base_allocations()
        accuracy_params = self.accuracy_parameters()

        if self.only_measured_profiles:
            throughput_params = self.throughput_parameters()
            latency_params = self.latency_parameters(only_measured_profiles=True)
            distinct_batches = sorted({
                batch
                for variants in throughput_params.values()
                for variant in variants.values()
                for batch in variant.keys()
            })
            distinct_batches = [batch for batch in distinct_batches if batch <= batching_cap]
        else:
            latency_params = self.latency_parameters(only_measured_profiles=False)

        # Define action and state spaces
        state_space = []
        for stage in stages:
            state_space.append({
                'variants': stages_variants[stage],
                'replicas': list(range(1, scaling_cap + 1)),
                'batches': distinct_batches,
            })

        # Initialize Q-table
        Q_table = {}

        # Set Q-Learning parameters
        learning_rate_q = 0.1
        discount_factor_q = 0.9
        exploration_rate = 1.0
        min_exploration_rate = 0.01
        exploration_decay_rate = 0.001
        max_steps_per_episode = 100

        num_episodes = num_state_limit  # Using num_state_limit as number of episodes

        for episode in range(num_episodes):
            # Initialize state randomly
            state = {}
            for stage in stages:
                state[stage] = {
                    'variant': random.choice(stages_variants[stage]),
                    'replica': random.randint(1, scaling_cap),
                    'batch': random.choice(distinct_batches),
                }
            for step in range(max_steps_per_episode):
                # Choose an action using epsilon-greedy policy
                if random.uniform(0, 1) > exploration_rate:
                    # Exploit: select the action with max Q value
                    state_key = tuple((stage, state[stage]['variant'], state[stage]['replica'],
                                       state[stage]['batch']) for stage in stages)
                    if state_key in Q_table and Q_table[state_key]:
                        action = {}
                        best_action = max(Q_table[state_key], key=Q_table[state_key].get)
                        for idx, stage in enumerate(stages):
                            action[stage] = {
                                'variant': best_action[idx][1],
                                'replica': best_action[idx][2],
                                'batch': best_action[idx][3],
                            }
                    else:
                        # If no action has been recorded for this state, choose randomly
                        action = {}
                        for stage in stages:
                            action[stage] = {
                                'variant':
                                random.choice(stages_variants[stage]),
                                'replica':
                                random.randint(max(1, state[stage]['replica'] - 1),
                                               min(scaling_cap, state[stage]['replica'] + 1)),
                                'batch':
                                random.choice(distinct_batches),
                            }
                else:
                    # Explore: select a random action
                    action = {}
                    for stage in stages:
                        action[stage] = {
                            'variant':
                            random.choice(stages_variants[stage]),
                            'replica':
                            random.randint(max(1, state[stage]['replica'] - 1),
                                           min(scaling_cap, state[stage]['replica'] + 1)),
                            'batch':
                            random.choice(distinct_batches),
                        }
                # Apply action to get next state
                next_state = {}
                for stage in stages:
                    next_state[stage] = {
                        'variant': action[stage]['variant'],
                        'replica': action[stage]['replica'],
                        'batch': action[stage]['batch'],
                    }
                # Apply configurations to the pipeline
                for task_id, stage in enumerate(stages):
                    self.pipeline.inference_graph[task_id].model_switch(
                        next_state[stage]['variant'])
                    self.pipeline.inference_graph[task_id].re_scale(
                        replica=next_state[stage]['replica'])
                    self.pipeline.inference_graph[task_id].change_batch(
                        batch=next_state[stage]['batch'])

                # Check constraints
                constraints_satisfied = self.constraints(arrival_rate=arrival_rate)

                if not constraints_satisfied:
                    reward = -1000  # Large negative reward for violating constraints
                else:
                    # Compute objectives
                    obj = self.objective(alpha=alpha, beta=beta, gamma=gamma)
                    reward = obj['objective']

                # Update Q-table
                state_key = tuple((stage, state[stage]['variant'], state[stage]['replica'],
                                   state[stage]['batch']) for stage in stages)
                action_key = tuple((stage, action[stage]['variant'], action[stage]['replica'],
                                    action[stage]['batch']) for stage in stages)
                next_state_key = tuple((stage, next_state[stage]['variant'],
                                        next_state[stage]['replica'], next_state[stage]['batch'])
                                       for stage in stages)

                if state_key not in Q_table:
                    Q_table[state_key] = {}

                if action_key not in Q_table[state_key]:
                    Q_table[state_key][action_key] = 0

                # Estimate Q(s, a)
                current_q = Q_table[state_key][action_key]

                # Get max Q for next state
                if next_state_key in Q_table and Q_table[next_state_key]:
                    max_future_q = max(Q_table[next_state_key].values())
                else:
                    max_future_q = 0

                # Update Q value
                new_q = current_q + learning_rate_q * (reward + discount_factor_q * max_future_q -
                                                       current_q)
                Q_table[state_key][action_key] = new_q

                # Update state
                state = next_state

                # Decay exploration rate
                exploration_rate = min_exploration_rate + (1.0 - min_exploration_rate) * np.exp(
                    -exploration_decay_rate * episode)

        # After training, extract the optimal policies
        # For each state, select the action with the highest Q-value
        optimal_states = []
        for state_key in Q_table:
            best_action = max(Q_table[state_key], key=Q_table[state_key].get)
            optimal_state = {}
            for idx, stage in enumerate(stages):
                optimal_state[stage] = {
                    'variant': best_action[idx][1],
                    'replica': best_action[idx][2],
                    'batch': best_action[idx][3],
                }
            # Apply configurations to the pipeline
            for task_id, stage in enumerate(stages):
                self.pipeline.inference_graph[task_id].model_switch(
                    optimal_state[stage]['variant'])
                self.pipeline.inference_graph[task_id].re_scale(
                    replica=optimal_state[stage]['replica'])
                self.pipeline.inference_graph[task_id].change_batch(
                    batch=optimal_state[stage]['batch'])

            # Check constraints
            if not self.constraints(arrival_rate=arrival_rate):
                continue

            # Generate state data
            state_data = {}
            if self.complete_profile:
                for task_id_j, task in enumerate(self.pipeline.inference_graph):
                    state_data.update({
                        f"task_{task_id_j}_latency": task.latency,
                        f"task_{task_id_j}_throughput": task.throughput,
                        f"task_{task_id_j}_throughput_all_replicas": task.throughput_all_replicas,
                        f"task_{task_id_j}_accuracy": task.accuracy,
                        f"task_{task_id_j}_measured": task.measured,
                        f"task_{task_id_j}_cpu_all_replicas": task.cpu_all_replicas,
                        f"task_{task_id_j}_gpu_all_replicas": task.gpu_all_replicas,
                    })
                state_data.update({
                    "pipeline_accuracy": self.pipeline.pipeline_accuracy,
                    "pipeline_latency": self.pipeline.pipeline_latency,
                    "pipeline_throughput": self.pipeline.pipeline_throughput,
                    "pipeline_cpu": self.pipeline.pipeline_cpu,
                    "pipeline_gpu": self.pipeline.pipeline_gpu,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "accuracy_objective": self.accuracy_objective(),
                    "resource_objective": self.resource_objective(),
                    "batch_objective": self.batch_objective(),
                })

            for task_id_j, task in enumerate(self.pipeline.inference_graph):
                state_data.update({
                    f"task_{task_id_j}_variant": task.active_variant,
                    f"task_{task_id_j}_cpu": task.cpu,
                    f"task_{task_id_j}_gpu": task.gpu,
                    f"task_{task_id_j}_batch": task.batch,
                    f"task_{task_id_j}_replicas": task.replicas,
                })

            state_data.update(self.objective(alpha=alpha, beta=beta, gamma=gamma))
            optimal_states.append(state_data)

            if len(optimal_states) >= num_state_limit:
                break

        return pd.DataFrame(optimal_states)

    def brute_force(
        self,
        scaling_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        arrival_rate: int,
        num_state_limit: int = None,
    ) -> pd.DataFrame:
        """Perform brute-force optimization to find the optimal state.

        Args:
            scaling_cap (int): Maximum number of allowed horizontal scaling for each node.
            alpha (float): Weight for accuracy objective.
            beta (float): Weight for resource usage objective.
            gamma (float): Weight for batch size objective.
            arrival_rate (int): Arrival rate into the pipeline.
            num_state_limit (int, optional): Limit on the number of states to evaluate.
                Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the optimal states.
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
        max_objective = states["objective"].max()
        optimal = states[states["objective"] == max_objective]
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
        """Optimize pipeline configuration using Gurobi solver.

        Args:
            scaling_cap (int): Maximum number of allowed horizontal scaling for each node.
            batching_cap (int): Maximum batch size allowed.
            alpha (float): Weight for accuracy objective.
            beta (float): Weight for resource usage objective.
            gamma (float): Weight for batch size objective.
            arrival_rate (int): Arrival rate into the pipeline.
            num_state_limit (int): Number of optimal states to retrieve.
            dir_path (str, optional): Directory path to save the model. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the optimal states.
        """
        self.only_measured_profiles = True  # Handle both cases using pre-calculated profiles
        sla = self.pipeline.sla
        stages = self.pipeline.stage_wise_task_names
        stages_variants = self.pipeline.stage_wise_available_variants

        # Retrieve parameters
        base_allocations = self.base_allocations()
        accuracy_params = self.accuracy_parameters()

        if self.only_measured_profiles:
            distinct_batches, throughput_params = self.throughput_parameters()
            latency_params = self.latency_parameters(only_measured_profiles=True)
            distinct_batches = [batch for batch in distinct_batches if batch <= batching_cap]
        else:
            latency_params = self.latency_parameters(only_measured_profiles=False)

        # Initialize Gurobi model
        model = gp.Model("pipeline")

        # Define variables
        variant_vars = model.addVars([(stage, variant) for stage in stages
                                      for variant in stages_variants[stage]],
                                     name="i",
                                     vtype=GRB.BINARY)

        replica_vars = model.addVars(stages, name="n", vtype=GRB.INTEGER, lb=1, ub=scaling_cap)

        if self.only_measured_profiles:
            batch_vars = model.addVars([(stage, batch) for stage in stages
                                        for batch in distinct_batches],
                                       name="b",
                                       vtype=GRB.BINARY)
            aux_batch_vars = model.addVars([(stage, variant, batch) for stage in stages
                                            for variant in stages_variants[stage]
                                            for batch in distinct_batches],
                                           name="aux",
                                           vtype=GRB.BINARY)
        else:
            batch_vars = model.addVars(stages, name="b", vtype=GRB.INTEGER, lb=1, ub=batching_cap)
            # Enforce batch sizes to be powers of 2
            batch_sizes = [2**i for i in range(int(math.log2(batching_cap)) + 1)]
            batch_size_indicator = model.addVars([(stage, size) for stage in stages
                                                  for size in batch_sizes],
                                                 vtype=GRB.BINARY,
                                                 name="batch_size_indicator")

        model.update()

        # Define helper functions
        def latency_function(batch: int, params: Dict[str, float]) -> float:
            """Calculate latency using a quadratic model."""
            coefficients = params["coefficients"]
            intercept = params["intercept"][0]
            return (coefficients[2] * (batch**2) + coefficients[1] * batch + coefficients[0] +
                    intercept)

        def queueing_latency(batch: int, arrival_rate: int) -> float:
            """Calculate queueing latency based on batch size and arrival rate."""
            return 0.0 if arrival_rate == 0 else (batch - 1) / arrival_rate

        # Add constraints
        if self.only_measured_profiles:
            # Throughput constraints
            for stage in stages:
                for variant in stages_variants[stage]:
                    for batch in distinct_batches:
                        model.addGenConstrAnd(
                            aux_batch_vars[stage, variant, batch],
                            [variant_vars[stage, variant], batch_vars[stage, batch]],
                            f"andconstr-{stage}-{variant}-{batch}")
                        model.addConstr(aux_batch_vars[stage, variant, batch] == 1 >>
                                        (replica_vars[stage] *
                                         throughput_params[stage][variant][batch] >= arrival_rate),
                                        name=f"throughput-{stage}-{variant}-{batch}")
            # Latency constraint
            total_latency = gp.quicksum(latency_params[stage][variant][batch] *
                                        variant_vars[stage, variant] * batch_vars[stage, batch] +
                                        queueing_latency(batch, arrival_rate) for stage in stages
                                        for variant in stages_variants[stage]
                                        for batch in distinct_batches)
            model.addConstr(total_latency <= sla, name="latency")

            # Ensure only one batch size is selected per stage
            for stage in stages:
                model.addConstr(gp.quicksum(batch_vars[stage, batch]
                                            for batch in distinct_batches) == 1,
                                name=f"single-batch-{stage}")
        else:
            # Throughput constraints
            for stage in stages:
                for variant in stages_variants[stage]:
                    latency_upper = self.pipeline_latency_upper_bound(stage, variant)
                    model.addConstr(
                        arrival_rate *
                        latency_function(batch_vars[stage], latency_params[stage][variant]) -
                        replica_vars[stage] * batch_vars[stage]
                        <= (arrival_rate * latency_upper - 1) * (1 - variant_vars[stage, variant]),
                        name=f"throughput-{stage}-{variant}")
            # Latency constraint
            total_latency = gp.quicksum(
                latency_function(batch_vars[stage], latency_params[stage][variant]) *
                variant_vars[stage, variant] + queueing_latency(batch_vars[stage], arrival_rate)
                for stage in stages for variant in stages_variants[stage])
            model.addConstr(total_latency <= sla, name="latency")

            # Enforce batch sizes to be powers of 2
            for stage in stages:
                model.addConstr(gp.quicksum(batch_size_indicator[stage, size]
                                            for size in batch_sizes) == 1,
                                name=f"one-batch-size-{stage}")
                for size in batch_sizes:
                    model.addConstr(batch_vars[stage]
                                    >= size - (max(batch_sizes) - min(batch_sizes)) *
                                    (1 - batch_size_indicator[stage, size]),
                                    name=f"batch-lb-{stage}-{size}")
                    model.addConstr(batch_vars[stage]
                                    <= size + (max(batch_sizes) - min(batch_sizes)) *
                                    (1 - batch_size_indicator[stage, size]),
                                    name=f"batch-ub-{stage}-{size}")

        # Baseline mode constraints
        if self.baseline_mode == "scale":
            for task in self.pipeline.inference_graph:
                model.addConstr(variant_vars[task.name, task.active_variant] == 1,
                                name=f"only-scale-{task.name}")
        elif self.baseline_mode == "switch":
            for task in self.pipeline.inference_graph:
                model.addConstr(replica_vars[task.name] == task.replicas,
                                name=f"only-switch-{task.name}")
        elif self.baseline_mode == "switch-scale":
            # TODO: Implement switch-scale baseline constraints
            pass

        # Ensure only one variant is active per stage
        for stage in stages:
            model.addConstr(gp.quicksum(variant_vars[stage, variant]
                                        for variant in stages_variants[stage]) == 1,
                            name=f"one_variant-{stage}")

        # Define objectives
        if self.pipeline.accuracy_method == "multiply":
            if len(stages) <= 2:
                accuracy_objective = gp.LinExpr()
                for stage in stages:
                    for variant in stages_variants[stage]:
                        accuracy_objective += accuracy_params[stage][variant] * variant_vars[
                            stage, variant]
                accuracy_objective = gp.quicksum([
                    accuracy_params[stage][variant] * variant_vars[stage, variant]
                    for stage in stages for variant in stages_variants[stage]
                ])
            else:
                # Handle multiplication of accuracies for up to three stages
                all_combinations = list(
                    itertools.product(*[stages_variants[stage] for stage in stages]))
                combination_vars = model.addVars(all_combinations, name="comb_i", vtype=GRB.BINARY)
                accuracy_obj = model.addVar(name="accuracy_objective",
                                            vtype=GRB.CONTINUOUS,
                                            lb=0,
                                            ub=1)
                model.addConstr(gp.quicksum(combination_vars[comb]
                                            for comb in all_combinations) == 1,
                                name='one_model_comb')
                for comb in all_combinations:
                    model.addConstr(combination_vars[comb] ==
                                    1 >> gp.quicksum(variant_vars[stages[i], comb[i]]
                                                     for i in range(len(stages))) == len(stages),
                                    name=f"combination_{comb}")
                    comb_accuracy = math.prod(
                        [accuracy_params[stages[i]][comb[i]] for i in range(len(stages))])
                    model.addConstr(combination_vars[comb] * comb_accuracy == accuracy_obj,
                                    name=f"accuracy_comb_{comb}")
                accuracy_objective = accuracy_obj
        elif self.pipeline.accuracy_method == "sum":
            accuracy_objective = gp.quicksum(accuracy_params[stage][variant] *
                                             variant_vars[stage, variant] for stage in stages
                                             for variant in stages_variants[stage])
        elif self.pipeline.accuracy_method == "average":
            accuracy_objective = gp.quicksum(
                (accuracy_params[stage][variant] * variant_vars[stage, variant]) / len(stages)
                for stage in stages for variant in stages_variants[stage])
        else:
            raise ValueError(f"Invalid accuracy method {self.pipeline.accuracy_method}")

        resource_objective = gp.quicksum(base_allocations[stage][variant] * replica_vars[stage] *
                                         variant_vars[stage, variant] for stage in stages
                                         for variant in stages_variants[stage])

        if self.only_measured_profiles:
            batch_objective = gp.quicksum(batch * batch_vars[stage, batch] for stage in stages
                                          for batch in distinct_batches)
        else:
            batch_objective = gp.quicksum(batch_vars[stage] for stage in stages)

        # Set objective
        model.setObjective(
            alpha * accuracy_objective - beta * resource_objective - gamma * batch_objective,
            GRB.MAXIMIZE)

        # Configure model parameters
        model.Params.PoolSearchMode = 2
        model.Params.LogToConsole = 0
        model.Params.PoolSolutions = num_state_limit
        model.Params.PoolGap = 0.0
        model.Params.NonConvex = 2  # Enable non-convex quadratic constraints

        model.update()

        # Optimize the model
        model.optimize()

        if dir_path:
            model.write(os.path.join(dir_path, "unmeasured.lp"))

        # Extract solutions
        states = []
        for sol_num in range(model.SolCount):
            model.Params.SolutionNumber = sol_num
            solution = model.getVars()
            var_values = {var.varName: var.Xn for var in solution}

            # Extract variant selections
            i_output = {}
            for var_name, value in var_values.items():
                if var_name.startswith("i[") and value > 0.5:
                    # Extract stage and variant from var_name, e.g., 'i[stage,variant]'
                    parts = var_name[2:-1].split(",")
                    if len(parts) == 2:
                        stage, variant = parts
                        i_output[stage.strip()] = variant.strip()

            # Extract replica counts
            n_output = {stage: int(var_values.get(f"n[{stage}]", 1)) for stage in stages}

            # Extract batch sizes
            if self.only_measured_profiles:
                b_output = {}
                for stage in stages:
                    for batch in distinct_batches:
                        if var_values.get(f"b[{stage},{batch}]", 0) > 0.5:
                            b_output[stage] = batch
                            break
            else:
                b_output = {stage: int(var_values.get(f"b[{stage}]", 1)) for stage in stages}

            # Apply configurations to the pipeline
            for task_id, stage in enumerate(stages):
                self.pipeline.inference_graph[task_id].model_switch(
                    i_output.get(stage, "default_variant"))
                self.pipeline.inference_graph[task_id].re_scale(replica=n_output.get(stage, 1))
                self.pipeline.inference_graph[task_id].change_batch(batch=b_output.get(stage, 1))

            # Generate state data
            state = {}
            if self.complete_profile:
                for task_id_j, task in enumerate(self.pipeline.inference_graph):
                    state.update({
                        f"task_{task_id_j}_latency": task.latency,
                        f"task_{task_id_j}_throughput": task.throughput,
                        f"task_{task_id_j}_throughput_all_replicas": task.throughput_all_replicas,
                        f"task_{task_id_j}_accuracy": task.accuracy,
                        f"task_{task_id_j}_measured": task.measured,
                        f"task_{task_id_j}_cpu_all_replicas": task.cpu_all_replicas,
                        f"task_{task_id_j}_gpu_all_replicas": task.gpu_all_replicas,
                    })
                state.update({
                    "pipeline_accuracy": self.pipeline.pipeline_accuracy,
                    "pipeline_latency": self.pipeline.pipeline_latency,
                    "pipeline_throughput": self.pipeline.pipeline_throughput,
                    "pipeline_cpu": self.pipeline.pipeline_cpu,
                    "pipeline_gpu": self.pipeline.pipeline_gpu,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "accuracy_objective": self.accuracy_objective(),
                    "resource_objective": self.resource_objective(),
                    "batch_objective": self.batch_objective(),
                })

            for task_id_j, task in enumerate(self.pipeline.inference_graph):
                state.update({
                    f"task_{task_id_j}_variant": task.active_variant,
                    f"task_{task_id_j}_cpu": task.cpu,
                    f"task_{task_id_j}_gpu": task.gpu,
                    f"task_{task_id_j}_batch": task.batch,
                    f"task_{task_id_j}_replicas": task.replicas,
                })

            state.update(self.objective(alpha=alpha, beta=beta, gamma=gamma))
            states.append(state)

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
        """Select and execute the optimization method.

        Args:
            optimization_method (str): The optimization method to use
                ('brute-force', 'gurobi' or 'q-learning').
            scaling_cap (int): Maximum number of allowed horizontal scaling for each node.
            alpha (float): Weight for accuracy objective.
            beta (float): Weight for resource usage objective.
            gamma (float): Weight for batch size objective.
            arrival_rate (int): Arrival rate into the pipeline.
            num_state_limit (int, optional): Number of optimal states to retrieve.
                Defaults to None.
            batching_cap (int, optional): Maximum batch size allowed for Gurobi. Required if
                method is 'gurobi'.
            dir_path (str, optional): Directory path to save the model. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the optimal states.
        """
        if optimization_method == "brute-force":
            optimal = self.brute_force(
                scaling_cap=scaling_cap,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit,
            )
        elif optimization_method == "gurobi":
            if batching_cap is None:
                raise ValueError("batching_cap must be provided for Gurobi optimization.")
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
            if batching_cap is None:
                raise ValueError("batching_cap must be provided for Q-Learning optimization.")
            optimal = self.q_learning_optimizer(
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
        """Check if the existing configuration can sustain the given load.

        Args:
            arrival_rate (int): The incoming load rate.

        Returns:
            bool: True if the load can be sustained, False otherwise.
        """
        return all(arrival_rate <= task.throughput_all_replicas
                   for task in self.pipeline.inference_graph)

    def sla_is_met(self) -> bool:
        """Check if the pipeline's latency meets the SLA.

        Returns:
            bool: True if SLA is met, False otherwise.
        """
        return self.pipeline.pipeline_latency < self.pipeline.sla

    def find_load_bottlenecks(self, arrival_rate: int) -> List[int]:
        """Identify tasks that are bottlenecks for the given load.

        Args:
            arrival_rate (int): The incoming load rate.

        Raises:
            ValueError: If the load can be sustained by all tasks.

        Returns:
            List[int]: List of task indices that are bottlenecks.
        """
        if self.can_sustain_load(arrival_rate=arrival_rate):
            raise ValueError("The load can be sustained! No bottleneck detected.")
        return [
            task_id for task_id, task in enumerate(self.pipeline.inference_graph)
            if arrival_rate > task.throughput_all_replicas
        ]

    def get_one_answer(self) -> Dict:
        """Retrieve one optimal configuration.

        Returns:
            Dict: A dictionary representing one optimal state.
        """
        pass
