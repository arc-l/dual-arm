import os
import matplotlib.pyplot as plt
import numpy as np
import re

results_folder = "results"
execution_times = {}

# Regex to extract number of objects and optional index
pattern = re.compile(r'_([0-9]+)_objects(?:_([0-9]+))?')

# Step 1: Read all duration files
for filename in os.listdir(results_folder):
    if filename.endswith("_duration.txt"):
        parts = filename.rsplit("_", 2)
        testcase = parts[0]  # e.g., task_random_6_objects_1
        method = parts[1]    # ours or baseline

        match = pattern.search(testcase)
        if not match:
            continue
        num_objects = int(match.group(1))
        index = int(match.group(2)) if match.group(2) is not None else 0

        # Read the float execution time
        with open(os.path.join(results_folder, filename), "r") as f:
            exec_time = float(f.read().strip())

        # Store under testcase name
        if testcase not in execution_times:
            execution_times[testcase] = {'num_objects': num_objects, 'index': index}
        execution_times[testcase][method] = exec_time

# Step 2: Filter only testcases with both methods
valid_cases = {
    name: data for name, data in execution_times.items()
    if "ours" in data and "baseline" in data
}

# Step 3: Sort by (num_objects, index)
sorted_items = sorted(
    valid_cases.items(),
    key=lambda item: (item[1]['num_objects'], item[1]['index'])
)

# Step 4: Prepare data for plotting
testcase_names = [name for name, _ in sorted_items]
ours_times = [data["ours"] for _, data in sorted_items]
baseline_times = [data["baseline"] for _, data in sorted_items]

# Step 5: Plot
x = np.arange(len(testcase_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, baseline_times, width, label="Baseline", color='gray')
ax.bar(x + width/2, ours_times, width, label="Ours", color='blue')

ax.set_xlabel("Test Cases")
ax.set_ylabel("Execution Time (s)")
ax.set_title("Execution Time Comparison: Ours vs Baseline")
ax.set_xticks(x)
ax.set_xticklabels(testcase_names, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("results/execution_time_comparison.png", dpi=500)
plt.show()
