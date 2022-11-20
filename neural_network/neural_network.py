import json
import csv
import sys
from typing import NamedTuple
from random import uniform
import math
from inspect import signature

TrainingData = NamedTuple(
    "TrainingData", [("input", tuple[float]), ("output", tuple[float])]
)
Vector = list[float]
Matrix = list[Vector]

def print_matrix(matrix: Matrix):
    print("\n".join(" ".join(f"{el:8.4f}" for el in row) for row in matrix))

def matrix_weight_mult(matrix: Matrix, vector: Vector):
    """returns matrix*vector. the vector is interpreted as a column matrix, so
    the dimensions of the result for an mxn matrix are mx1 (and the vector must
    have n elements.)"""
    assert len(matrix[0]) == len(vector), \
        "number of cols in matrix must match number of elements in vector"
    return [
        sum(matrix[row][col]*vector[col] for col in range(len(vector)))
        for row in range(len(matrix))
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
    object_params = signature(NeuralNetwork.__init__).parameters
    config_types = (
        {k: object_params[k].annotation for k in object_params if k != "self" and k != "_"} |
        {"training_data_location": str}
    )
    config_items = set(config_types.keys())
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
    return config | {k: config_types[k](input(k+": ")) for k in needed_items}

def create_random_matrix(
    rows: int, cols: int, lower_bound: float = -1.0, upper_bound: float = 1.0
) -> Matrix:
    return [
        [uniform(lower_bound, upper_bound) for _ in range(cols)]
        for _ in range(rows)
    ]

def compute_error(target_output: Vector, actual_output: Vector):
    return math.dist(target_output, actual_output)

class NeuralNetwork:
    def __init__(self, input_variables: int, output_variables: int, hidden_layers: int,
        neurons_per_hidden_layer: int, neuron_bias: float, neuron_firing_threshold: float,
        max_cycles: int, learning_error_threshold: float, learning_rate: float, **_
    ):
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.hidden_layers = hidden_layers
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.neuron_bias = neuron_bias
        self.neuron_firing_threshold = neuron_firing_threshold
        self.max_cycles = max_cycles
        self.learning_error_threshold = learning_error_threshold
        self.learning_rate = learning_rate
        self.layer_matrices = self.create_initial_matrices()

    def create_initial_matrices(self) -> list[Matrix]:
        """There is one matrix for the input layer that produces its output, which
        is the input for the first hidden layer; then num_hidden_layers matrices
        that produce the output for each successive hidden layer, with the final one
        dimensionally configured to create num_outputs values."""
        return [
            create_random_matrix(self.neurons_per_hidden_layer, self.input_variables),
            *(
                create_random_matrix(
                    self.neurons_per_hidden_layer, self.neurons_per_hidden_layer
                )
                for _ in range(self.hidden_layers-1)
            ),
            create_random_matrix(self.output_variables, self.neurons_per_hidden_layer)
        ]

    def threshold_fire(self, incoming_value: float) -> float:
        if incoming_value > self.neuron_firing_threshold:
            return 1 / (1 + math.exp(-incoming_value))
        else:
            return 0

    def process_network_layer(self, input_weights: Matrix, input_vector: Vector) -> Vector:
        weighted_and_biased = [
            d+self.neuron_bias for d in 
            matrix_weight_mult(input_weights, input_vector)
        ]
        return [self.threshold_fire(d) for d in weighted_and_biased]
    
    def process_input(self, input: Vector) -> Vector:
        assert len(input) == self.input_variables, "incorrect number of inputs given"
        output = input
        for matrix in self.layer_matrices:
            output = self.process_network_layer(matrix, output)
        return [self.threshold_fire(v) for v in output]

    def train_network(self, training_data: list[TrainingData]) -> list[Matrix]:
        for i in range(self.max_cycles):
            cycle_test_data = training_data[i % len(training_data)]
            layer_one_input = cycle_test_data.input
            desired_output = cycle_test_data.output
            layer_one_input = map(lambda d: d + self.bias, layer_one_input)
            output = [self.threshold_fire(d) for d in layer_one_input]
            for layer_matrix in self.layer_matrices:
                output = self.process_network_layer(layer_matrix, output)
            assert len(output) == self.output_variables
            error = math.dist(output, desired_output)
            print("error:", error)
            if error < self.learning_error_threshold:
                print("error dropped below threshold")
                return self.layer_matrices
            else:
                pass  # aaaa
        print("max cycles reached")
        return self.layer_matrices
    
    def test_network(self, data: list[TrainingData]) -> float:
        error = 0
        for test in data:
            output = self.process_input(test.input)
            error += compute_error(test.output, output)
        return error/len(data)


if __name__ == "__main__":
    config = read_config()
    print("configuration:", config)
    network = NeuralNetwork(**config)
    training_data = read_training_data(
        config["training_data_location"],
        config["input_variables"],
        config["output_variables"]
    )
    testing_data = training_data[:len(training_data)//4]
    training_data = training_data[len(training_data)//4:]
    print(
        f"average error on {len(testing_data)} samples with no training:",
        network.test_network(testing_data)
    )
    
