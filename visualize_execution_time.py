import os
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

data_folder = "data"
results_folder = "results"
execution_times_by_type = {}
ignore_baseline_for_double_cycled = True

# Define fixed order of testcase types
testcase_types_order = ["mixed", "cycled", "double_cycled", "random"]
methods = ["ours", "baseline", "baseline_with_our_tp", "ours_with_baseline_tp", "our_tp_with_curobo", "baseline_tp_with_curobo"]
method_display_names = {
    "ours": "Ours",
    "baseline": "Baseline",
    "baseline_with_our_tp": "Baseline+\nOur TP",
    "ours_with_baseline_tp": "Ours+\nBaseline TP",
    "our_tp_with_curobo": "Our TP+\nCurobo",
    "baseline_tp_with_curobo": "Baseline TP+\nCurobo"
}

colors = {
    "baseline": "gray",
    "ours": "blue",
    "baseline_with_our_tp": "green",
    "ours_with_baseline_tp": "orange",
    "our_tp_with_curobo": "purple",
    "baseline_tp_with_curobo": "brown"
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
ax.bar(tp_x, tp_times, width, label="Baseline+Our TP", color=colors["baseline_with_our_tp"])
ax.bar(ours_tp_x, ours_tp_times, width, label="Ours+Baseline TP", color=colors["ours_with_baseline_tp"])
ax.bar(our_tp_with_curobo_x, our_tp_with_curobo_times, width, label="Our TP+Curobo", color=colors["our_tp_with_curobo"])
ax.bar(ours_x, ours_times, width, label="Ours", color=colors["ours"])


ax.set_xlabel("Test Cases", fontname='Times New Roman', fontsize=20)
ax.set_ylabel("Execution Time (s)", fontname='Times New Roman', fontsize=20)
ax.set_title("Execution Time Comparison: Ours vs Baselines", fontname='Times New Roman', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(testcase_names, rotation=45, ha="right", fontname='Times New Roman', fontsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.xaxis.set_tick_params(labelsize=20)
leg = ax.legend(fontsize=20, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5)
for text in leg.get_texts():
    text.set_fontname('Times New Roman')
    text.set_fontsize(20)
plt.tight_layout()
plt.savefig("results/execution_time_comparison.png", dpi=100)
print("Plot saved as results/execution_time_comparison.png")
# plt.show()

# Step 5: Success Rate Plot
total_cases = len(sorted_cases)
success_counts = {method: 0 for method in methods}

for case in sorted_cases:
    for method in methods:
        if ignore_baseline_for_double_cycled and case["name"].startswith("double_cycled") and method == "baseline":
            continue  # don't penalize baseline for this
        if case[method] > 0:  # treat >0 as success
            success_counts[method] += 1

success_rates = {method: success_counts[method] / total_cases for method in methods}

# Plot success rate
fig, ax = plt.subplots(figsize=(10, 5))
bar_x = np.arange(len(methods))
bar_heights = [success_rates[method] for method in methods]
bar_colors = [colors[method] for method in methods]

ax.bar(bar_x, bar_heights, color=bar_colors, tick_label=[method_display_names[m] for m in methods])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Success Rate", fontname='Times New Roman', fontsize=20)
ax.set_title("Success Rate by Method", fontname='Times New Roman', fontsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)

# Annotate bar values
for i, rate in enumerate(bar_heights):
    # Place annotation inside the bar, with smaller font and white color
    ax.text(i, rate - 0.03, f"{rate:.2f}", ha='center', va='top', fontname='Times New Roman', fontsize=20, color='white')

plt.tight_layout()
plt.savefig("results/success_rate_comparison.png", dpi=100)
print("Plot saved as results/success_rate_comparison.png")

# Step 6: Execution Time Ratios vs Ours
ratio_data = {method: [] for method in methods if method != "ours"}

for case in sorted_cases:
    ours_time = case["ours"]
    if ours_time <= 0:
        continue  # skip cases where ours failed

    for method in ratio_data:
        method_time = case[method]
        if method_time > 0:
            ratio = method_time / ours_time
            ratio_data[method].append(ratio)

# Box plot data
labels = list(ratio_data.keys())
data = [ratio_data[method] for method in labels]
box_colors = [colors[method] for method in labels]
labels = [method_display_names[m] for m in labels]

fig, ax = plt.subplots(figsize=(10, 5))
box = ax.boxplot(data, patch_artist=True, labels=labels, medianprops=dict(color='black', linewidth=2))

# Color each box
for patch, color in zip(box['boxes'], box_colors):
    patch.set_facecolor(color)

ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1)

ax.set_ylabel("Execution Time Ratio", fontname='Times New Roman', fontsize=20)
ax.set_title("Execution Time Ratio Compared to Ours", fontname='Times New Roman', fontsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.tight_layout()
plt.savefig("results/execution_time_ratios_boxplot.png", dpi=100)
print("Plot saved as results/execution_time_ratios_boxplot.png")
# plt.show()
