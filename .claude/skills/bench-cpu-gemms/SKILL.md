---
name: bench-cpu-gemms
description: 'This skill is for benchmarking CPU GEMM performance for shapes and precisions used in the model. It will run the benchmark for each shape and precision and print the performance in TFLOPs and bandwidth in GB/s.'
---

# Step 1: Read a model configration and get shapes

- Read the model configuration file from the huggingface directory ($HF_HOME/hub/) and get the shapes of the GEMM operations in the model. 
- We basically want to get N, K for each GEMM operation in the model.
- Vary the M parameter for each GEMM operation in the model. Assume, M is equivalent to batch size and can be any values in these ranges: 1, 2, 4, 8, 16, 32, 64, 128, 256. 512.
- Attempt to ascertain the shapes of GEMM operations for TP=1 and TP=2.
- Also attempt to get the precision of the GEMM operations in the model.
- Write these shapes into a references/gemm_shapes.txt file. This file will be used as input for the benchmark script in the next step.

# Step 2: Reinstall the environment

- In this step you will reinstall the environment to ensure that you have a clean setup for benchmarking CPU GEMMs. 
  You will use the scripts/install_script.sh script of this skill to reinstall the updated code and dependencies.
  This script will install all the necessary dependencies for benchmarking CPU GEMMs.


# Step 3: Run the benchmark with environment

- Run the benchmark scripts using the scripts/bench_environ.sh from this skill. 
- Read the shapes from the references/gemm_shapes.txt file and run the benchmark script for each shape.
- Benchmark scripts for each precision are present in the scripts directory of this skill. They take M, N, K as input, run the benchmark and print the performance in TFLOPs and bandwidth in GB/s. 
- Write the shape, precision, TFLOPs and bandwidth into a references/gemm_benchmark_results.csv file.

# Step 4: Plot the results

- Plot the results from the references/gemm_benchmark_results.csv file using the scripts/plot_gemm_results.py script. This script will generate plots for each precision and save them in the references/plots directory. The plots will show the performance in TFLOPs and bandwidth in GB/s for each shape and precision.

