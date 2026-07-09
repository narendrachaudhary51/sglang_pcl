---
name: bench-cpu-moe
description: 'This skill is for benchmarking fused CPU MoE kernel performance for shapes and precisions used in a model. It will run the benchmark for each shape and precision and print the performance in TFLOPs and bandwidth in GB/s.'
---

# Prestep: Acquire and navigate to compute node

- Acquire a compute node with the following command:
```bash
  salloc --partition=emr --constraint="ddr5600" --time=03:59:00
```
- Use the following command to navigate to the compute node:
```bash
  srun --pty /bin/bash
```
# Step 1: Read a model configration and get shapes

- Read the model configuration file from the huggingface directory ($HF_HOME/hub/) and get the shapes of the MoE operations in the model. 
- We basically want to get N, K, num_experts and topk parameters needed for fused MoE kernel.
- Also attempt to get the precision of the MoE operations in the model.
- Attempt to ascertain the shapes of MoE operations for TP=1 and TP=2.
- Write these shapes into a references/${model_name}/moe_shapes.txt file. This file will be used as input for the benchmark script in the next step.

# Step 2: Edit TILE SIZES for GEMM operations

- In this step, we edit lines 7-9 of sgl-kernel/csrc/cpu/gemm.h file to set the TILE sizes for MoE operations.
- Pick a TILE size configration (TILE_M, TILE_N, TILE_K) from tile_config.yaml in this directory.  
- Record the configrations. We will rename the plot and csv files to include the TILE sizes in the file names. This will help us to identify 
  the performance of MoE operations for different TILE sizes.

# Step 3: Rebuild and Reinstall the environment

- In this step you will reinstall the environment to ensure that you have a clean setup for benchmarking CPU MoE kernels. 
  You will use the scripts/install_script.sh script of this skill to reinstall the updated code and dependencies.
  This script will install all the necessary dependencies for benchmarking CPU MoE operations.


# Step 4: Run the benchmark with environment

- Run the benchmark scripts using the scripts/bench_environ.sh from this skill. 
- Vary the M parameter for each MoE operation in the model. Assume, M is equivalent to batch size and can be any values in these ranges: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512.
- Read the shapes from the N and K parameters from references/${model_name}/moe_shapes.txt file to get the data for each shape.
- Run the scripts/run_moe_sweep.py with M parameter list as input. This script will run the benchmark for each shape and precision and write the results into a references/${model_name}/moe_benchmark_results.csv file. 
- Benchmark functions and scripts for each precision are present in the scripts directory of this skill. They take M, N, K as input, run the benchmark and return time in ms, performance in TFLOPs and bandwidth in GB/s. 
- Write the shape, precision, TFLOPs and bandwidth into a references/${model_name}/moe_benchmark_results.csv file.

# Step 5: Plot the results

- Plot the results from the references/${model_name}/moe_benchmark_results.csv file using the scripts/${model_name}/plot_moe_results.py script. This script takes model name as an input. This script will generate plots for each precision and save them in the references/plots directory. The plots will show the performance in TFLOPs and bandwidth in GB/s for each shape and precision.

# Step 6: Rename the plots and csv files

- Rename the plots and csv files to include the TILE sizes in the file names. This will help us to identify the performance of MoE operations for different TILE sizes. For example, if the TILE sizes are TILE_M=32, TILE_N=32, TILE_K=32, then the plot and csv files will be renamed to moe_benchmark_results_TILE_M_32_TILE_N_32_TILE_K_32.csv and moe_benchmark_results_TILE_M_32_TILE_N_32_TILE_K_32.png respectively.

# Step 7: Repeat steps 2-6 for different tile size configrations 