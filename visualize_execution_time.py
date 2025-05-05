import os
import matplotlib.pyplot as plt
import numpy as np

data_folder = "data"
results_folder = "results"
execution_times_by_type = {}

# Define fixed order of testcase types
testcase_types_order = ["mixed", "cycled", "double_cycled", "random"]

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
    ours_file = f"{testcase_name}_ours_duration.txt"
    baseline_file = f"{testcase_name}_baseline_duration.txt"

    ours_path = os.path.join(results_folder, ours_file)
    baseline_path = os.path.join(results_folder, baseline_file)

    # Default values
    ours_time = 0.0
    baseline_time = 0.0

    if not os.path.exists(ours_path):
        print(f"[Missing ours]     {ours_file}")
    else:
        try:
            with open(ours_path, "r") as f:
                ours_time = float(f.read().strip())
        except ValueError:
            print(f"[Parse error ours] {ours_file}")

    if not os.path.exists(baseline_path):
        print(f"[Missing baseline] {baseline_file}")
    else:
        try:
            with open(baseline_path, "r") as f:
                baseline_time = float(f.read().strip())
        except ValueError:
            print(f"[Parse error baseline] {baseline_file}")

    # Determine testcase type
    testcase_type = next((t for t in testcase_types_order if testcase_name.startswith(t)), "other")

    num_objects = extract_num_objects(testcase_name)

    if testcase_type not in execution_times_by_type:
        execution_times_by_type[testcase_type] = []

    execution_times_by_type[testcase_type].append({
        "name": testcase_name,
        "ours": ours_time,
        "baseline": baseline_time,
        "num_objects": num_objects
    })

# Step 2: Sort within each type and collect in final list
sorted_cases = []
for t_type in testcase_types_order:
    if t_type in execution_times_by_type:
        sorted_cases.extend(
            sorted(execution_times_by_type[t_type], key=lambda x: x["num_objects"])
        )

# Step 3: Prepare data for plotting
testcase_names = [case["name"] for case in sorted_cases]
ours_times = [case["ours"] for case in sorted_cases]
baseline_times = [case["baseline"] for case in sorted_cases]

# Step 4: Plot
x = np.arange(len(testcase_names))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width / 2, baseline_times, width, label="Baseline", color='gray')
ax.bar(x + width / 2, ours_times, width, label="Ours", color='blue')

ax.set_xlabel("Test Cases")
ax.set_ylabel("Execution Time (s)")
ax.set_title("Execution Time Comparison: Ours vs Baseline")
ax.set_xticks(x)
ax.set_xticklabels(testcase_names, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("results/execution_time_comparison.png", dpi=500)
print("Plot saved as results/execution_time_comparison.png")
plt.show()
