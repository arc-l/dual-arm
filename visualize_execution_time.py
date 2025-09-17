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
    "ours": "SDAR",
    "baseline": "Baseline",
    "baseline_with_our_tp": "SDAR-T+\nBaseline MP",
    "ours_with_baseline_tp": "Baseline TP+\nSDAR-M",
    "our_tp_with_curobo": "SDAR-T+\nCurobo",
    "baseline_tp_with_curobo": "Baseline TP+\nCurobo"
}

colors = {
    "baseline": "gray",
    "ours": "blue",
    "baseline_with_our_tp": "green",
    "ours_with_baseline_tp": "orange",
    "our_tp_with_curobo": "purple",
    "baseline_tp_with_curobo": "magenta"
}

type_short = {
    "mixed": "M",
    "cycled": "S",
    "double_cycled": "D",
    "random": "R",
    "other": "O"
}

# Helper to extract number of objects from testcase name
def extract_num_objects(name):
    parts = name.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0

def get_type_short(name):
    parts = name.split('_')
    if len(parts) >= 2 and parts[0] == 'double' and parts[1] == 'cycled':
        return 'D'
    elif parts[0] == 'mixed':
        return 'M'
    elif parts[0] == 'cycled':
        return 'S'
    elif parts[0] == 'random':
        return 'R'
    else:
        return 'O'

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
testcase_names = [f"{get_type_short(case['name'])}{case['num_objects']}" for case in sorted_cases]
n_cases = len(testcase_names)
x = np.linspace(0, n_cases - 1, n_cases)
width = 0.15  # Keep bars thick

# Bar positions
baseline_x = x - 2.5 * width
tp_x = x - 1.5 * width
ours_tp_x = x - 0.5 * width
our_tp_with_curobo_x = x + 0.5 * width
baseline_tp_with_curobo_x = x + 1.5 * width
ours_x = x + 2.5 * width



# Get times
baseline_times = [case["baseline"] for case in sorted_cases]
baseline_tp_with_curobo_times = [case["baseline_tp_with_curobo"] for case in sorted_cases]
tp_times = [case["baseline_with_our_tp"] for case in sorted_cases]
ours_tp_times = [case["ours_with_baseline_tp"] for case in sorted_cases]
our_tp_with_curobo_times = [case["our_tp_with_curobo"] for case in sorted_cases]
ours_times = [case["ours"] for case in sorted_cases]


# Step 4: Plot
fig, ax = plt.subplots(figsize=(28, 6))
ax.margins(x=0)
ax.bar(baseline_x, baseline_times, width, label="Baseline", color=colors["baseline"])
ax.bar(tp_x, tp_times, width, label="SDAR-T+Baseline", color=colors["baseline_with_our_tp"])
ax.bar(ours_tp_x, ours_tp_times, width, label="Baseline TP+SDAR-M", color=colors["ours_with_baseline_tp"])
ax.bar(our_tp_with_curobo_x, our_tp_with_curobo_times, width, label="SDAR-T+Curobo", color=colors["our_tp_with_curobo"])
ax.bar(baseline_tp_with_curobo_x, baseline_tp_with_curobo_times, width, label="Baseline TP+Curobo", color=colors["baseline_tp_with_curobo"])
ax.bar(ours_x, ours_times, width, label="SDAR", color=colors["ours"])


# ax.set_xlabel("Test Cases", fontname='Times New Roman', fontsize=20)
ax.set_ylabel("Execution Time (s)", fontname='Times New Roman', fontsize=25)
ax.set_xticks(x)
ax.set_xlim(-1, np.max(x) + 1)
ax.set_xticklabels(testcase_names, rotation=45, fontname='Times New Roman', fontsize=25)
ax.yaxis.set_tick_params(labelsize=25)
ax.xaxis.set_tick_params(labelsize=25)
leg = ax.legend(fontsize=25, loc='upper center', ncol=6)
for text in leg.get_texts():
    text.set_fontname('Times New Roman')
    text.set_fontsize(25)
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
fig, ax = plt.subplots(figsize=(10, 4))
bar_x = np.arange(len(methods))
bar_heights = [success_rates[method] for method in methods]
bar_colors = [colors[method] for method in methods]

ax.bar(bar_x, bar_heights, color=bar_colors, tick_label=[method_display_names[m] for m in methods])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Success Rate", fontname='Times New Roman', fontsize=20)
ax.xaxis.set_tick_params(labelsize=17)
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
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.tight_layout()
plt.savefig("results/execution_time_ratios_boxplot.png", dpi=100)
print("Plot saved as results/execution_time_ratios_boxplot.png")
# plt.show()
