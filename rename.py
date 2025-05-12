
import os

root_dir = './outputs/minisubset02/'  # replace with your actual path

for person_folder in os.listdir(root_dir):
    person_path = os.path.join(root_dir, person_folder)
    if not os.path.isdir(person_path):
        continue

    for filename in os.listdir(person_path):
        if filename.endswith('.json') and 'deepseek-r1_prompt_' in filename:
            old_path = os.path.join(person_path, filename)
            new_filename = filename.replace('deepseek-r1_prompt_', 'deepseek-r1-llama-70B_prompt_')
            new_path = os.path.join(person_path, new_filename)
            os.rename(old_path, new_path)
            print(f'Renamed: {old_path} → {new_path}')
