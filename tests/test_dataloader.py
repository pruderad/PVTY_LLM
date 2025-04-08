from data_loader import DataLoader
import json

dataloader = DataLoader('data_json/data_minisubset04_02.json')

persons_list = dataloader.load_persons_from_json()



for person in persons_list:
    print(person.name)
    print(person.captions)
    print(person.text)

    print("..............................")

print(len(persons_list))
