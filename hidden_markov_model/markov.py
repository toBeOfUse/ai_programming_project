import sys
import json

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("argument for transitions and emissions required")
    else:
        with open(sys.argv[1]) as transitionfile:
            transitions: dict = json.load(transitionfile)
        print("loaded transition matrix")
        print("states are:", list(transitions.keys()))
        for key in transitions:
            # data validation
            assert list(transitions.keys()) == list(transitions[key].keys())
            assert sum(transitions[key].values()) == 1.0
        # transitions["A"]["B"] gives the probability of the transition from A to B
