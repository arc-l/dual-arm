from task import Task
import os


if __name__ == "__main__":
    data_dir = "data"
    task_files = [
        "random_objects_18.json",
        "cycled_objects_12.json",
        "double_cycled_12.json",
        "mixed_15.json"
    ]
    for task_file in task_files:
        task = Task.read_from_json(f"{data_dir}/{task_file}")
        task.draw_all(os.path.join("results", task_file.replace('.json', '.png')))