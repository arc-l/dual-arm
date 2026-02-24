import os, shutil
import re
from task import Task, generate_dataset

object_size = 0.045
table_size = 0.6

def generate_random():
    with open("README.md", "a") as f:
        f.write("## Random Objects\n")
        for i in range(4, 19):
            task = Task(i, table_size, table_size, object_size)
            try:
                task.generate(gen_type="random", clearance=0.2)
            except Exception as e:
                print("clearance 0.1 failed, trying clearance 0.1")
                try:
                    task.generate(gen_type="random", clearance=0.1)
                except Exception as e:
                    print("clearance 0.1 failed, trying clearance 0.05")
                    try:
                        task.generate(gen_type="random", clearance=0.05)
                    except Exception as e:
                        print("clearance 0.05 failed, trying clearance 0.0")
                        task.generate(gen_type="random", clearance=0.0)

            task.write_to_json(f"data/random_objects_{i}.json")
            task.draw_all(f"data_vis/random_objects_{i}.png")

def generate_cycled():
    with open("README.md", "a") as f:
        f.write("## Cycled Objects\n")
        for i in range(4, 13):
            task = Task(i, table_size, table_size, object_size)
            try:
                task.generate(gen_type="cycled", clearance=0.2)
            except Exception as e:
                print("clearance 0.1 failed, trying clearance 0.1")
                try:
                    task.generate(gen_type="cycled", clearance=0.1)
                except Exception as e:
                    print("clearance 0.1 failed, trying clearance 0.05")
                    try:
                        task.generate(gen_type="cycled", clearance=0.05)
                    except Exception as e:
                        print("clearance 0.05 failed, trying clearance 0.0")
                        task.generate(gen_type="cycled", clearance=0.0)
            task.write_to_json(f"data/cycled_objects_{i}.json")
            task.draw_all(f"data_vis/cycled_objects_{i}.png")

def generate_double_cycled():
    with open("README.md", "a") as f:
        f.write("## Double Cycled Objects\n")
        for i in range(6, 13):
            task = Task(i, table_size, table_size, object_size)
            try:
                task.generate(gen_type="double_cycle", clearance=0.2)
            except Exception as e:
                print("clearance 0.1 failed, trying clearance 0.1")
                try:
                    task.generate(gen_type="double_cycle", clearance=0.1)
                except Exception as e:
                    print("clearance 0.1 failed, trying clearance 0.05")
                    try:
                        task.generate(gen_type="double_cycle", clearance=0.05)
                    except Exception as e:
                        print("clearance 0.05 failed, trying clearance 0.0")
                        task.generate(gen_type="double_cycle", clearance=0.0)
            task.write_to_json(f"data/double_cycled_{i}.json")
            task.draw_all(f"data_vis/double_cycled_{i}.png")

def generate_mixed():
    with open("README.md", "a") as f:
        f.write("## Mixed Objects\n")
        for i in range(16):
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
            task.draw_all(f"data_vis/mixed_{i}.png")

if __name__ == "__main__":
    data_folder = "data"
    data_vis_folder = "data_vis"
    # debug the whole process
    for folder in [data_folder, data_vis_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            # remove the folder and create it again
            shutil.rmtree(folder)
            os.makedirs(folder)

    generate_random()
    generate_cycled()
    generate_double_cycled()
    generate_mixed()

