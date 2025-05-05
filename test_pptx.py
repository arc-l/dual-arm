import os, shutil
import re
from task import Task, generate_dataset

object_size = 0.045
table_size = 0.6


def generate_mixed(i):
    task = Task(i, table_size, table_size, object_size)
    try:
        task.generate(gen_type="mixed", clearance=0.2)
    except Exception as e:
        print("clearance 0.1 failed, trying clearance 0.1")
        try:
            task.generate(gen_type="mixed", clearance=0.1)
        except Exception as e:
            print("clearance 0.1 failed, trying clearance 0.05")
            try:
                task.generate(gen_type="mixed", clearance=0.05)
            except Exception as e:
                print("clearance 0.05 failed, trying clearance 0.0")
                task.generate(gen_type="mixed", clearance=0.0)
    task.write_to_json(f"data/mixed_{i}.json")
    task.draw_four_panel(f"data_vis/mixed_{i}.png")

if __name__ == "__main__":
    generate_mixed(5)