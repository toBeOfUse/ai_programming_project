import json
import csv
import sys
from collections import namedtuple
from random import uniform

TrainingData = namedtuple("TrainingData", ["input", "output"])
Matrix = list[list[float]]

def print_matrix(matrix: Matrix):
    print("\n".join(" ".join(f"{el:8.4f}" for el in row) for row in matrix))

def read_training_data(
    file_path: str, num_inputs: int, num_outputs: int
) -> list[TrainingData]:
    with open(file_path) as data_file:
        reader = csv.reader(data_file)
        data = []
        for row in reader:
            assert len(row) == num_inputs + num_outputs
            data.append(
                TrainingData(
                    input=tuple(map(float, row[:num_inputs])),
                    output=tuple(map(float, row[-num_outputs:]))
                )
            )
    return data

def read_config() -> dict[str, int|float|str]:
    config_items = {'input_variables', 'output_variables', 'hidden_layers', 
        'perceptrons_per_hidden_layer', 'perceptron_bias',
        'perceptron_firing_threshold', 'max_cycles', 'learning_threshold', 
        'learning_rate', 'training_data_location'}
    config = {}
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1]) as config_file:
                config = json.load(config_file)
        except:
            print(f"failed to read command-line argument config file {sys.argv[1]}")
    else:
        print("config file not provided as command line argument")
    needed_items = config_items.difference(config.keys())
    if len(needed_items) > 0:
        print("please enter the needed neural network configuration options:")
    return config | {k: float(input(k)) for k in needed_items}

def create_random_matrix(
    rows: int, cols: int, lower_bound: float = -1.0, upper_bound: float = 1.0
) -> Matrix:
    return [
        [uniform(lower_bound, upper_bound) for _ in range(cols)]
        for _ in range(rows)
    ]

def create_initial_matrices(
    num_inputs: int, num_outputs: int, num_hidden_layers: int, num_perceptrons: int
) -> list[Matrix]:
    return [
        create_random_matrix(num_inputs, num_perceptrons),
        *(
            create_random_matrix(num_perceptrons, num_perceptrons)
            for _ in range(num_hidden_layers)
        ),
        create_random_matrix(num_perceptrons, num_outputs)
    ]

def train_network() -> list[Matrix]:
    config = read_config()
    initial_matrices = create_initial_matrices(
        config["input_variables"], config["output_variables"], config["hidden_layers"],
        config["perceptrons_per_hidden_layer"]
    )
    training_data = read_training_data(
        config["training_data_location"], config["input_variables"], config["output_variables"]
    )
    for m in initial_matrices:
        print_matrix(m)
        print()
    print(training_data)

if __name__ == "__main__":
    train_network()
