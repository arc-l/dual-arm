import os
from task import Task 

# Main script to go through all .json task files
data_folder = "data"
save_folder = "data_vis"
for file in os.listdir(data_folder):
    if file.endswith(".json"):
        json_path = os.path.join(data_folder, file)
        output_path = os.path.join(save_folder, file.replace(".json", ".png"))
        try:
            task = Task.read_from_json(json_path)
            task.draw_triptych(output_path)
        except Exception as e:
            print(f"Failed to process {file}: {e}")
