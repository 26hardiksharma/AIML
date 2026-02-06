import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "data.txt")
def load_data(filename):
    with open(file_path,"r") as f:
        data = json.load(f)
    return data
data = load_data(file_path)

def clean_data(data):
    text_to_num = {
        "one":"1",
        "two":"2",
        "three":"3",
        "four":"4",
        "five":"5"
    }
    cleaned_data = []
    unique = set()
    for user in data:
        raw_rating = str(user["rating"]).strip().lower()
        if(raw_rating in text_to_num):
            user["rating"] = text_to_num[raw_rating]
        raw_age = user.get("age")
        if(raw_age == None):
            user["age"] = None
        if(user["name"] in unique):
            continue
        cleaned_data.append(user)
    return cleaned_data
data = clean_data(data)
for i in data:
    print(i)