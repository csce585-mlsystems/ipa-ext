# Experiment Setup Discussion

## Key Challenges

- Understanding Chameleon, Trace, and Zeus
- Establishing the standard way of monitoring energy
- Integrating energy profiling with the existing code base
- Adapting the workload to the new variable (and possible improvements)

## Initial Experiment

- Replication of the existing IPA setup
- Plot existing workload accuracy, latency, and cost
- Set up an `etrace2` shell script with Zeus comments inside Python files
- Plot CPU energy and GPU energy

## Metrics

- **Accuracy**
- **Latency**
- **Cost**
- **Energy**
  - CPU
  - GPU

## Design of Experiments (Initial Sketch)

### Independent Variables

- Model size
- Batch size
- Hardware configuration
- Workload types (steady vs. bursty)

### Dependent Variables

- Energy consumption
- Accuracy
- Latency
- Cost

### Control Variables

- Existing IPA experiments/workloads
- Inference pipelines
- Baselines (similar systems)
- Measurement tools and methods
- Data preprocessing steps
- Duration and number of inference runs
- Resource allocation policies
- Evaluation metrics (end-to-end evaluation)

## General Discussion

- Discussed work distribution
- All team members are starting with replication to gain an understanding of all the components of the paper
- The next step will be to use `etrace` and Zeus to gather energy information and plot it over the existing IPA paper workloads
