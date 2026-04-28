import json

FILE = "interview_state.json"


def save_state(state):
    with open(FILE, "w") as f:
        json.dump(state, f)


def load_state():
    try:
        with open(FILE, "r") as f:
            return json.load(f)
    except:
        return {}