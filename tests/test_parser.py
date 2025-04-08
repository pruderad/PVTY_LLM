from parser import PersonParser
import json

def compare_json_files(file1: str, file2: str) -> bool:
    """
    Compares two JSON files for equality, ignoring key order.
    Args:
        file1 (str): Path to the first JSON file.
        file2 (str): Path to the second JSON file.
    Returns:
        bool: True if the JSON content is identical, False otherwise.
    """
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    return data1 == data2

if __name__ == "__main__":
    parser = PersonParser('./datasets/minisubset04_annotated')
    parser.parse_all_persons(path='./data_json/data_minisubset04_03.json', write=True)


    file1 = './data_json/data_minisubset04_03.json'
    file2 = './data_json/data_minisubset04_03.json'
    if compare_json_files(file1, file2):
        print("The JSON files are identical.")
    else:
        print("The JSON files are different.")