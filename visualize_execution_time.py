import os
import matplotlib.pyplot as plt
import numpy as np

data_folder = "data"
results_folder = "results"
execution_times_by_type = {}
ignore_baseline_for_double_cycled = True

# Define fixed order of testcase types
testcase_types_order = ["mixed", "cycled", "double_cycled", "random"]
methods = ["ours", "baseline", "baseline_with_our_tp", "ours_with_baseline_tp", "our_tp_with_curobo"]
colors = {
    "baseline": "gray",
    "ours": "blue",
    "baseline_with_our_tp": "green",
    "ours_with_baseline_tp": "orange",
    "our_tp_with_curobo": "purple",
}

# Helper to extract number of objects from testcase name
def extract_num_objects(name):
    parts = name.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0

# Step 1: Process each JSON testcase
for filename in os.listdir(data_folder):
    if not filename.endswith(".json"):
        continue
    testcase_name = filename[:-5]  # Remove ".json"

    # Determine testcase type
    testcase_type = next((t for t in testcase_types_order if testcase_name.startswith(t)), "other")
    num_objects = extract_num_objects(testcase_name)

    result_entry = {"name": testcase_name, "num_objects": num_objects}

    for method in methods:
        duration_file = f"{testcase_name}_{method}_duration.txt"
        duration_path = os.path.join(results_folder, duration_file)

        exec_time = 0.0
        if not os.path.exists(duration_path):
            if not (testcase_name.startswith("double_cycled") and method == "baseline"):
                print(f"[Missing {method}] {duration_file}")
        else:
            try:
                with open(duration_path, "r") as f:
                    exec_time = float(f.read().strip())
            except ValueError:
                print(f"[Parse error {method}] {duration_file}")

        if ignore_baseline_for_double_cycled and testcase_name.startswith("double_cycled") and method == "baseline":
            result_entry[method] = 0
        else:
            result_entry[method] = exec_time


    if testcase_type not in execution_times_by_type:
        execution_times_by_type[testcase_type] = []

    execution_times_by_type[testcase_type].append(result_entry)

# Step 2: Sort within each type and collect into final list
sorted_cases = []
for t_type in testcase_types_order:
    if t_type in execution_times_by_type:
        sorted_cases.extend(
            sorted(execution_times_by_type[t_type], key=lambda x: x["num_objects"])
        )

# Step 3: Prepare data for plotting
testcase_names = [case["name"] for case in sorted_cases]
group_spacing = 1.2  # or try 1.5 for even more spacing
x = np.arange(len(testcase_names)) * group_spacing
width = 0.2

# Bar positions
baseline_x = x - 2 * width
tp_x = x - width
ours_tp_x = x
our_tp_with_curobo_x = x + width
ours_x = x + 2 * width



# Get times
baseline_times = [case["baseline"] for case in sorted_cases]
tp_times = [case["baseline_with_our_tp"] for case in sorted_cases]
ours_tp_times = [case["ours_with_baseline_tp"] for case in sorted_cases]
our_tp_with_curobo_times = [case["our_tp_with_curobo"] for case in sorted_cases]
ours_times = [case["ours"] for case in sorted_cases]


# Step 4: Plot
fig, ax = plt.subplots(figsize=(18, 6))
ax.bar(baseline_x, baseline_times, width, label="Baseline", color=colors["baseline"])
ax.bar(tp_x, tp_times, width, label="Baseline w/ Our TP", color=colors["baseline_with_our_tp"])
ax.bar(ours_tp_x, ours_tp_times, width, label="Ours w/ Baseline TP", color=colors["ours_with_baseline_tp"])
ax.bar(our_tp_with_curobo_x, our_tp_with_curobo_times, width, label="Our TP w/ Curobo", color=colors["our_tp_with_curobo"])
ax.bar(ours_x, ours_times, width, label="Ours", color=colors["ours"])


ax.set_xlabel("Test Cases")
ax.set_ylabel("Execution Time (s)")
ax.set_title("Execution Time Comparison: Ours vs Baselines")
ax.set_xticks(x)
ax.set_xticklabels(testcase_names, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("results/execution_time_comparison.png", dpi=500)
print("Plot saved as results/execution_time_comparison.png")
# plt.show()
