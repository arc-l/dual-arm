import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

results_folder = "results"
method = "ours"
colors = {
    "ours": "blue"
}
time_fields = ["filter_task_plan_time", "task_planning_time", "motion_planning_time"]
labels = ["Filter Task Plan", "Task Planning", "Motion Planning"]
map_types = ["mixed", "cycled", "double_cycled", "random", "other"]

# Helper to infer map type from filename
def infer_map_type(fname):
    for mt in map_types:
        if fname.startswith(mt):
            return mt
    return "other"

# Collect stats by map type and by testcase
stats_by_type = {mt: {field: [] for field in time_fields} for mt in map_types}
all_case_stats = []

for fname in os.listdir(results_folder):
    if not fname.endswith("time_stats.json"):
        continue
    if f"_{method}_" not in fname:
        continue
    mt = infer_map_type(fname)
    with open(os.path.join(results_folder, fname), "r") as f:
        data = json.load(f)
    case_stats = {"name": fname, "map_type": mt}
    for field in time_fields:
        vals = data.get(field, [])
        if isinstance(vals, list):
            stats_by_type[mt][field].extend(vals)
            case_stats[field] = np.mean(vals) if vals else 0
        elif isinstance(vals, (int, float)):
            stats_by_type[mt][field].append(vals)
            case_stats[field] = vals
        else:
            case_stats[field] = 0
    all_case_stats.append(case_stats)

# Plot for all test cases (each bar is a testcase)
fig, ax = plt.subplots(figsize=(max(12, len(all_case_stats)*0.5), 6))
case_names = [c["name"].replace("_ours_time_stats.json","") for c in all_case_stats]
bar_width = 0.5
for i, field in enumerate(time_fields):
    vals = [c[field] for c in all_case_stats]
    ax.bar(np.arange(len(case_names)) + i*bar_width/len(time_fields), vals, bar_width/len(time_fields), label=labels[i])
ax.set_xticks(np.arange(len(case_names)) + bar_width/3)
ax.set_xticklabels(case_names, rotation=45, ha='right', fontname='Times New Roman', fontsize=10)
ax.set_ylabel("Average Time (s)", fontname='Times New Roman', fontsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.xaxis.set_tick_params(labelsize=10)
leg = ax.legend(fontsize=16, loc='upper right')
for text in leg.get_texts():
    text.set_fontname('Times New Roman')
    text.set_fontsize(16)
plt.tight_layout()
out_path = os.path.join(results_folder, "planning_time_by_testcase.png")
plt.savefig(out_path, dpi=100)
print(f"Plot saved as {out_path}")

# Plot all map types in one grouped bar plot with different colors
map_type_colors = {
    "mixed": "red",
    "cycled": "green",
    "double_cycled": "blue",
    "random": "orange",
    "other": "gray"
}
x = np.arange(len(time_fields))
width = 0.18
fig, ax = plt.subplots(figsize=(10, 6))
for i, mt in enumerate(map_types):
    averages = [np.mean(stats_by_type[mt][field]) if stats_by_type[mt][field] else 0 for field in time_fields]
    ax.bar(x + i*width, averages, width, label=mt.capitalize(), color=map_type_colors.get(mt, "gray"))
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(labels, fontname='Times New Roman', fontsize=20)
ax.set_ylabel("Average Time (s)", fontname='Times New Roman', fontsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.xaxis.set_tick_params(labelsize=20)
leg = ax.legend(fontsize=16, loc='upper right')
for text in leg.get_texts():
    text.set_fontname('Times New Roman')
    text.set_fontsize(16)
plt.tight_layout()
out_path = os.path.join(results_folder, "planning_time_comparison_all_types.png")
plt.savefig(out_path, dpi=100)
print(f"Plot saved as {out_path}")

# Pie plot for average time distribution across all fields (remove task planning part)
pie_stats = {field: [] for field in time_fields}
for fname in os.listdir(results_folder):
    if not fname.endswith("time_stats.json"):
        continue
    if f"_{method}_" not in fname:
        continue
    with open(os.path.join(results_folder, fname), "r") as f:
        data = json.load(f)
    filter_times = data.get("filter_task_plan_time", [])
    motion_times = data.get("motion_planning_time", [])
    pie_stats["filter_task_plan_time"].extend(filter_times if isinstance(filter_times, list) else [filter_times])
    pie_stats["motion_planning_time"].extend(motion_times if isinstance(motion_times, list) else [motion_times])

avg_times = [np.mean(pie_stats[field]) if pie_stats[field] else 0 for field in ["filter_task_plan_time", "motion_planning_time"]]
pie_labels = ["SelectBestPlan", "Motion Planning"]
total_avg = sum(avg_times)
if total_avg > 0:
    tab_colors = matplotlib.cm.get_cmap('tab10')(np.linspace(0, 1, 2))
    fig, ax = plt.subplots(figsize=(8, 4))
    wedges, texts, autotexts = ax.pie(
        avg_times,
        labels=pie_labels,
        autopct=lambda pct: f"{pct:.1f}%\n({pct*total_avg/100:.2f}s)",
        colors=tab_colors,
        textprops={'fontname': 'Times New Roman', 'fontsize': 18, 'color': 'black'},
        wedgeprops={'edgecolor': 'white'}
    )
    for autotext in autotexts:
        autotext.set_fontsize(18)
        autotext.set_color('black')
        autotext.set_weight('bold')
    # ax.set_title("Average Planning Time Distribution Per Step", fontname='Times New Roman', fontsize=22)
    plt.tight_layout()
    out_path = os.path.join(results_folder, "planning_time_pie.png")
    plt.savefig(out_path, dpi=100)
    print(f"Pie plot saved as {out_path}")
