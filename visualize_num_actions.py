import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt

"""visualize_num_actions.py

Reads all files ending with `_actions.json` inside the `results` folder.
Each file is expected to follow the naming pattern:
	<testcase_name>_<method>_actions.json

Where <method> is either `baseline` or `ours` (only those two are considered
for the ratio). The JSON can be either:
 - a list of length 2: [arm0_actions, arm1_actions]
 - a dict with string keys "0" and "1" mapping to each arm's action list
For each arm we count the number of non-null (not None) entries. The total
actions for a testcase+method is the sum over both arms, ignoring nulls.

For every testcase that has both baseline and ours data and ours > 0, it
computes the ratio: baseline / ours. Ratios < 1 are skipped (kept same
behavior as prior version).

It then produces:
1. A boxplot of baseline/ours ratios per map type.
2. (Optional) A bar plot of the mean ratio per map type with 95% CI error bars.

Outputs are saved into:
	results/num_actions_ratio_boxplot.png
	results/num_actions_ratio_means.png
	results/num_actions_absolute_barplot.png
"""


RESULTS_DIR = "results"
METHODS = ["baseline", "ours"]
MAP_TYPE_PREFIXES = ["mixed", "cycled", "double_cycled", "random"]


def infer_map_type(testcase_name: str) -> str:
	for prefix in MAP_TYPE_PREFIXES:
		if testcase_name.startswith(prefix):
			return prefix
	return "other"


def extract_testcase_and_method(filename: str):
	# Expect pattern: <testcase>_<method>_actions.json
	if not filename.endswith("_actions.json"):
		return None, None
	core = filename[:-len("_actions.json")]  # strip suffix
	for m in METHODS:
		if core.endswith(f"_{m}"):
			testcase = core[: -len(f"_{m}")]
			return testcase, m
	return None, None

def count_actions_from_json(path: str) -> int:
	"""Load an actions.json file and count non-null actions across both arms."""
	with open(path, "r") as f:
		data = json.load(f)
	arm_lists = []
	if isinstance(data, list):
		arm_lists = data[:2]
	elif isinstance(data, dict):
		# collect keys '0' and '1' if present
		arm_lists = [data.get("0", []), data.get("1", [])]
	count = 0
	for arm in arm_lists:
		if isinstance(arm, list):
			for a in arm:
				if a is not None:
					count += 1
	return count


def read_action_counts():
	data = defaultdict(dict)  # testcase -> method -> value
	if not os.path.isdir(RESULTS_DIR):
		raise FileNotFoundError(f"Results directory '{RESULTS_DIR}' not found.")

	for fname in os.listdir(RESULTS_DIR):
		if not fname.endswith("_actions.json"):
			continue
		testcase, method = extract_testcase_and_method(fname)
		if not testcase or method not in METHODS:
			continue
		path = os.path.join(RESULTS_DIR, fname)
		try:
			value = count_actions_from_json(path)
			data[testcase][method] = float(value)
		except Exception as e:
			print(f"[WARN] Failed reading {fname}: {e}")
	return data


def compute_ratios(data):
	# Returns map_type -> list of ratios (baseline/ours)
	ratios_by_type = defaultdict(list)
	per_testcase_ratios = {}
	for testcase, methods_dict in data.items():
		if all(m in methods_dict for m in METHODS):
			ours = methods_dict["ours"]
			baseline = methods_dict["baseline"]
			if ours > 0:
				ratio = baseline / ours if baseline >= 0 else float('nan')
				map_type = infer_map_type(testcase)
				ratios_by_type[map_type].append(ratio)
				per_testcase_ratios[testcase] = ratio
			else:
				print(f"[SKIP] Ours has non-positive actions for {testcase}, skipping ratio.")
		else:
			missing = [m for m in METHODS if m not in methods_dict]
			print(f"[MISS] {testcase} missing methods: {missing}")
	return ratios_by_type, per_testcase_ratios


def plot_boxplot(ratios_by_type):
	if not ratios_by_type:
		print("No ratios to plot.")
		return
	labels = []
	data = []
	for mt in MAP_TYPE_PREFIXES + ["other"]:
		if mt in ratios_by_type and ratios_by_type[mt]:
			labels.append(mt)
			data.append(ratios_by_type[mt])

	if not data:
		print("No ratio data lists to plot.")
		return

	plt.figure(figsize=(10, 5))
	box = plt.boxplot(data, patch_artist=True, labels=labels)
	colors = ["lightblue", "lightgreen", "violet", "wheat", "lightgray"]
	for patch, c in zip(box['boxes'], colors):
		patch.set_facecolor(c)
	plt.axhline(1.0, color='red', linestyle='--', linewidth=1)
	plt.ylabel("Num Actions Ratio (baseline / ours)")
	plt.title("Baseline vs Ours: Num Actions Ratio by Map Type")
	plt.tight_layout()
	out_path = os.path.join(RESULTS_DIR, "num_actions_ratio_boxplot.png")
	plt.savefig(out_path, dpi=400)
	print(f"Saved boxplot to {out_path}")


def plot_mean_bars(ratios_by_type):
	means = []
	labels = []
	for mt in MAP_TYPE_PREFIXES + ["other"]:
		vals = ratios_by_type.get(mt, [])
		if not vals:
			continue
		labels.append(mt)
		n = len(vals)
		mean = sum(vals) / n
		means.append(mean)

	if not means:
		return

	plt.figure(figsize=(8, 4))
	x = range(len(labels))
	plt.bar(x, means, color="steelblue")
	plt.axhline(1.0, color='red', linestyle='--', linewidth=1)
	plt.xticks(x, labels)
	plt.ylabel("Mean Ratio (baseline / ours)")
	plt.title("Mean Num Actions Ratio by Map Type")
	for i, m in enumerate(means):
		plt.text(i, m + 0.02, f"{m:.2f}", ha='center', va='bottom')
	plt.tight_layout()
	out_path = os.path.join(RESULTS_DIR, "num_actions_ratio_means.png")
	plt.savefig(out_path, dpi=400)
	print(f"Saved mean ratio bar plot to {out_path}")


def extract_num_objects(testcase_name: str) -> int:
	# Find last underscore followed by digits
	parts = testcase_name.split('_')
	for part in reversed(parts):
		if part.isdigit():
			return int(part)
	return -1


def plot_absolute_actions(action_counts):
	"""Grouped bar plot of absolute action counts per testcase (baseline vs ours)."""
	entries = []  # list of (map_type, num_objects, testcase, baseline_cnt, ours_cnt)
	for testcase, methods_dict in action_counts.items():
		if all(m in methods_dict for m in METHODS):
			map_type = infer_map_type(testcase)
			n_objs = extract_num_objects(testcase)
			entries.append((map_type, n_objs, testcase, methods_dict['baseline'], methods_dict['ours']))

	if not entries:
		print("No complete testcase pairs for absolute actions plot.")
		return

	# Sort by map type defined order then num objects then name
	order_index = {mt: i for i, mt in enumerate(MAP_TYPE_PREFIXES + ["other"])}
	entries.sort(key=lambda x: (order_index.get(x[0], 999), x[1], x[2]))

	testcase_names = [e[2] for e in entries]
	baseline_vals = [e[3] for e in entries]
	ours_vals = [e[4] for e in entries]

	import numpy as np
	x = np.arange(len(testcase_names))
	width = 0.4

	plt.figure(figsize=(18, 6))
	plt.bar(x - width/2, baseline_vals, width, label='Baseline', color='gray')
	plt.bar(x + width/2, ours_vals, width, label='Ours', color='blue')
	plt.ylabel('Number of Actions')
	plt.xlabel('Test Cases')
	plt.title('Absolute Number of Actions per Test Case')
	plt.xticks(x, testcase_names, rotation=45, ha='right')
	plt.legend()
	plt.tight_layout()
	out_path = os.path.join(RESULTS_DIR, 'num_actions_absolute_barplot.png')
	plt.savefig(out_path, dpi=400)
	print(f"Saved absolute actions bar plot to {out_path}")


def main():
	data = read_action_counts()
	print(f"Loaded action counts for {len(data)} testcases.")
	ratios_by_type, per_testcase = compute_ratios(data)
	if per_testcase:
		print("Sample ratios (first 10):")
		for k in list(per_testcase.keys())[:10]:
			print(f"  {k}: {per_testcase[k]:.3f}")
	plot_boxplot(ratios_by_type)
	plot_mean_bars(ratios_by_type)
	plot_absolute_actions(data)


if __name__ == "__main__":
	main()

