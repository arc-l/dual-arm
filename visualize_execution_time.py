import os
import matplotlib.pyplot as plt
import numpy as np

# Folder containing results
results_folder = "results"

# Dictionary to store execution times
execution_times = {}

# Read all txt files in the results folder
for filename in os.listdir(results_folder):
    if filename.endswith("_duration.txt"):
        parts = filename.rsplit("_", 2)  # Split by "_" keeping last two parts separate
        testcase = "_".join(parts[:-2])  # Extract testcase name
        method = parts[-2]  # Extract method name (ours or baseline)

        # Read the execution time from file
        with open(os.path.join(results_folder, filename), "r") as f:
            exec_time = float(f.read().strip())

        # Store in dictionary
        if testcase not in execution_times:
            execution_times[testcase] = {}
        execution_times[testcase][method] = exec_time

# Filter test cases that have results for both "ours" and "baseline"
valid_cases = {tc: data for tc, data in execution_times.items() if "ours" in data and "baseline" in data}

# Prepare data for plotting
testcases = list(valid_cases.keys())
ours_times = [valid_cases[tc]["ours"] for tc in testcases]
baseline_times = [valid_cases[tc]["baseline"] for tc in testcases]

# Plot results
x = np.arange(len(testcases))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, baseline_times, width, label="Baseline", color='gray')
ax.bar(x + width/2, ours_times, width, label="Ours", color='blue')

# Labeling
ax.set_xlabel("Test Cases")
ax.set_ylabel("Execution Time (s)")
ax.set_title("Comparison of Execution Time: Ours vs Baseline")
ax.set_xticks(x)
ax.set_xticklabels(testcases, rotation=45, ha="right")
ax.legend()

# Show the plot
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "execution_time_comparison.png"), dpi=500)
plt.show()
