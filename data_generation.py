import os, shutil
import re
from task import Task, generate_dataset

object_size = 0.045
table_size = 0.6

def generate_random():
    with open("README.md", "a") as f:
        f.write("## Random Objects\n")
        for i in range(4, 16):
            repeat = 1
            if 10 <= i <= 12:
                repeat = 2
            for r in range(repeat):
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

                task.write_to_json(f"data/random_objects_{i}_{r}.json")
                task.draw_four_panel(f"data_vis/random_objects_{i}_{r}.png")
                f.write(f"### {i} objects\n")
                f.write(f"![{i} objects](data_vis/random_objects_{i}_{r}.png)\n")

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
            task.draw_four_panel(f"data_vis/cycled_objects_{i}.png")
            f.write(f"### {i} objects\n")
            f.write(f"![{i} objects](data_vis/cycled_objects_{i}.png)\n")

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
            task.draw_four_panel(f"data_vis/double_cycled_{i}.png")
            f.write(f"### {i} objects\n")
            f.write(f"![{i} objects](data_vis/double_cycled_{i}.png)\n")

if __name__ == "__main__":
    # clean up the readme file
    readme_path = "README.md"
    with open(readme_path, "r") as f:
        content = f.read()
    
    # Remove everything after the comment
    cleaned = re.split(r'<!--\s*Start\s*-->', content)[0] + "\n<!-- Start -->\n"
    with open("README.md", "w") as f:
        f.write(cleaned)

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



