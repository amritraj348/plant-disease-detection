import json

def load_labels(path):
    with open(path, "r") as f:
        return json.load(f)
