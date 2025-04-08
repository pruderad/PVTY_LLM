import json
import os
from pathlib import Path
from dateutil import parser

def extract_year(date_str):
    try:
        dt = parser.parse(date_str, fuzzy=True)
        return dt.year
    except (ValueError, TypeError):
        return None  # Return None if parsing fails

def test_annotation(path: str, folder: str):
    folder_path = Path(path) / folder
    name = folder

    json_file_annotated = folder_path / "annotation.json"
    if not json_file_annotated.exists():
        return None, 0, 'not_annotated', name
    
    with open(json_file_annotated, "r", encoding="utf-8") as f:
        annotated_data = json.load(f)
    
    # Filter for valid annotations
    valid_annotations = [entry for entry in annotated_data if (entry['birthday_annotated'] or entry['birth_year'] != '') and entry['figure_year_annotated']]
    
    if not valid_annotations:
        return None, 0, 'not_fully_annotated', name
    
    json_file_annotated_LLM = folder_path / f"{name}_LLM_data.json"
    if not json_file_annotated_LLM.exists():
        return None, 0, 'not_LLM_annotated', name
    
    with open(json_file_annotated_LLM, "r", encoding="utf-8") as f:
        LLM_data = json.load(f)
    
    errors = []
    can_not_annotate = 0
    for annotated_entry, llm_entry in zip(annotated_data, LLM_data):
        if not (annotated_entry['birthday_annotated'] and annotated_entry['figure_year_annotated']):
            continue
        try:
            annotated_birth_year = int(annotated_entry['birth_year'])
            annotated_photo_year = int((int(annotated_entry['estimated_year_creation_right']) + int(annotated_entry['estimated_year_creation_left'])) / 2)
            
            LLM_birth_year_int = extract_year(llm_entry['birthday'])
            LLM_photo_year = llm_entry['year_of_photo_int']
            can_annotate = llm_entry['can_determine']
            caption = llm_entry['caption']
            
            if LLM_birth_year_int is None or LLM_photo_year is None:
                continue
            
            age_manually = annotated_photo_year - annotated_birth_year
            age_LLM = LLM_photo_year - LLM_birth_year_int
            if can_annotate:
                error = abs(age_manually - age_LLM)

                errors.append(error)
                if error:
                    print(f"Name: {name}")
                    print(f"Caption: {caption}")
                    print('Manually annotated:')
                    print(f"Annotated birth year: {annotated_birth_year}")
                    print(f"Annotated photo year: {annotated_photo_year}")
                    print(f"Age manually: {age_manually}")
                    print()
                    print('LLM annotated:')
                    print(f"LLM birth year: {LLM_birth_year_int}")
                    print(f"LLM photo year: {LLM_photo_year}")
                    print(f"Age LLM: {age_LLM}")
                    print(f"Error: {error}")
                    print('-------------------------------------')
            else:
                can_not_annotate += 1
                #print(caption)
        except (ValueError, TypeError, KeyError):
            continue
    
    return True, sum(errors), can_not_annotate, name

if __name__ == '__main__':
    subsetPath = Path(os.getcwd()) / "minisubset04_annotated"
    folders = [str(f.name) for f in subsetPath.iterdir() if f.is_dir()]
    folders.sort()
    
    tested = 0
    error_all = 0
    not_annotated = 0
    not_LLM_annotated = 0
    not_fully_annotated = 0
    can_not_annotate = 0
    for folder in folders:
        ret, error, reason, name = test_annotation(subsetPath, folder)
        if ret:
            can_not_annotate += reason
            if not reason:
                tested += 1
            error_all += error 
        
        else:
            if reason == 'not_annotated':
                not_annotated += 1
            elif reason == 'not_LLM_annotated':
                not_LLM_annotated += 1
            elif reason == 'not_fully_annotated':
                not_fully_annotated += 1
    
    print("All people: ", len(folders))
    print("Tested: ", tested)
    print('Not manually annotated: ', not_annotated)
    print('Not LLM annotated: ', not_LLM_annotated)
    print('Not fully manually annotated: ', not_fully_annotated)
    print('Can not determine: ', can_not_annotate)
    print("Total Error: ", error_all)