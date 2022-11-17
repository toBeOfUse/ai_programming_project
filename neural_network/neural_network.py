import json
import csv
import sys
from typing import NamedTuple
from random import uniform
import math

TrainingData = NamedTuple(
    "TrainingData", [("input", tuple[float]), ("output", tuple[float])]
)
Vector = list[float]
Matrix = list[Vector]

def print_matrix(matrix: Matrix):
    print("\n".join(" ".join(f"{el:8.4f}" for el in row) for row in matrix))

def matrix_weight_mult(matrix: Matrix, vector: Vector):
    assert len(matrix) == len(vector), \
        "number of rows in matrix must match number of elements in vector"
    return [
        sum(vector[row]*matrix[row][col] for row in range(len(vector)))
        for col in range(len(matrix[0]))
    ]

def read_training_data(
    file_path: str, num_inputs: int, num_outputs: int
) -> list[TrainingData]:
    with open(file_path) as data_file:
        reader = csv.reader(data_file)
        data = []
        for row in reader:
            assert len(row) == num_inputs + num_outputs, \
                "incorrect number of data points per row in test data"
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
        'perceptron_firing_threshold', 'max_cycles', 'learning_error_threshold', 
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
    # TODO: training_data_location should not be cast to float (and maybe others
    # should be int)
    return config | {k: float(input(k+": ")) for k in needed_items}

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

def threshold_fire(incoming_value: float, threshold: float) -> float:
    if incoming_value > threshold:
        return 1.0 / (1 + math.e**(-incoming_value))
    else:
        return 0.0

def process_network_layer(
    input_weights: Matrix, input_vector: Vector, bias: float, threshold: float
) -> Vector:
    weighted_and_biased = [d+bias for d in matrix_weight_mult(input_weights, input_vector)]
    return [threshold_fire(d, threshold) for d in weighted_and_biased]

def train_network() -> list[Matrix]:
    config = read_config()
    layer_matrices = create_initial_matrices(
        config["input_variables"], config["output_variables"], config["hidden_layers"],
        config["perceptrons_per_hidden_layer"]
    )
    training_data = read_training_data(
        config["training_data_location"], config["input_variables"], config["output_variables"]
    )
    bias = config["perceptron_bias"]
    threshold = config["perceptron_firing_threshold"]
    for i in range(config["max_cycles"]):
        cycle_test_data = training_data[i % len(training_data)]
        layer_one_input = cycle_test_data.input
        desired_output = cycle_test_data.output
        layer_one_input = map(lambda d: d + bias, layer_one_input)
        output = [threshold_fire(d, threshold) for d in layer_one_input]
        for layer_matrix in layer_matrices:
            output = process_network_layer(layer_matrix, output, bias, threshold)
        assert len(output) == config["output_variables"]
        error = math.dist(output, desired_output)

if __name__ == "__main__":
    train_network()
