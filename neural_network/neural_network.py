from __future__ import annotations
import json
import csv
import sys
from typing import NamedTuple
from random import uniform
import math
from inspect import signature
import numpy as np

TrainingData = NamedTuple(
    "TrainingData", [("input", tuple[float]), ("output", tuple[float])]
)

def print_matrix(matrix: np.array):
    print("\n".join(" ".join(f"{el:8.4f}" for el in row) for row in matrix))

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
    return config | {k: eval(config_types[k])(input(k+": ")) for k in needed_items}

def create_random_matrix(
    rows: int, cols: int, lower_bound: float = -1.0, upper_bound: float = 1.0
) -> np.array:
    return np.array([
        [uniform(lower_bound, upper_bound) for _ in range(cols)]
        for _ in range(rows)
    ])

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
    
    def cost_derivative(self, actual_output: np.array, target_output: np.array) -> np.array:
        if type(target_output) is tuple:
            target_output = np.array([target_output]).T
        return actual_output-target_output
    
    def activation(self, cumulative_input):
        return 1 / (1 + np.exp(-cumulative_input))
    
    def activation_derivative(self, cumulative_input):
        return self.activation(cumulative_input) * (1-self.activation(cumulative_input))

    def create_initial_matrices(self) -> list[np.array]:
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
        if abs(incoming_value) > self.neuron_firing_threshold:
            return self.activation(incoming_value)
        else:
            return incoming_value*0

    def process_network_layer(self, input_weights: np.array, input_vector: np.array) -> np.array:
        weighted_and_biased = np.dot(input_weights, input_vector) + self.neuron_bias
        return np.array([self.threshold_fire(d) for d in weighted_and_biased])
    
    def process_input(self, input: np.array, debug: bool=False) -> np.array:
        assert len(input) == self.input_variables, "incorrect number of inputs given"
        output = input
        if debug: print("input:\n", output)
        for matrix in self.layer_matrices:
            if debug: print("weights:\n",matrix)
            output = self.process_network_layer(matrix, output)
            if debug: print("layer result:\n",output)
        return output

    def train_network(self, training_data: list[TrainingData], debug_output: bool=False) -> None:
        for i in range(self.max_cycles):
            # process data forwards, accumulating the input and output values
            cycle_test_data = training_data[i % len(training_data)]
            input_layer_output = np.array([cycle_test_data.input]).T
            desired_output = cycle_test_data.output
            current_output = input_layer_output
            outputs = [np.array(input_layer_output)]
            cumulative_inputs = []
            # process all hidden layers to eventually get input to output layer
            for layer_matrix in self.layer_matrices:
                cumulative_input = np.dot(layer_matrix, current_output)+self.neuron_bias
                cumulative_inputs.append(cumulative_input)
                current_output = np.array([self.threshold_fire(d) for d in cumulative_input])
                outputs.append(current_output)
            assert len(current_output) == self.output_variables
            error = math.dist(tuple(current_output), desired_output)
            if debug_output:
                print(f"error with test data {cycle_test_data} is {error}")
            if error < self.learning_error_threshold:
                if debug_output:
                    print("error below learning threshold")
                continue
            else:
                activation_derivative = self.activation_derivative(cumulative_inputs[-1])
                cost_gradient_vector = (
                    self.cost_derivative(outputs[-1], desired_output) *
                    activation_derivative
                )
                derivatives_per_weight = np.dot(cost_gradient_vector, outputs[-2].T)
                weight_deltas = [derivatives_per_weight]
                #as we go backward we accumulate weight deltas for each layer in reverse
                for adjusting_layer_index in range(2, self.hidden_layers+2):
                    cumulative_input = outputs[-adjusting_layer_index]
                    activation_derivative = self.activation_derivative(cumulative_input)
                    cost_gradient_vector = np.dot(
                        self.layer_matrices[-adjusting_layer_index+1].T,
                        cost_gradient_vector
                    ) * activation_derivative
                    weight_deltas.insert(
                        0,
                        np.dot(cost_gradient_vector, outputs[-adjusting_layer_index-1].T)
                    )
                    if debug_output:
                        print("changes in weights:")
                        print(weight_deltas)
                self.layer_matrices = [
                    weight-self.learning_rate*delta_weight
                    for weight, delta_weight in zip(self.layer_matrices, weight_deltas)
                ]

        print("max cycles reached")
        return
    
    def test_network(self, data: list[TrainingData], debug: bool = False) -> float:
        "returns average error for testing data (euclidean distance)"
        error = 0
        for test in data:
            output = self.process_input(test.input, debug)
            error += math.dist(test.output, output)
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
    testing_data = training_data[:len(training_data)//5]
    training_data = training_data[len(training_data)//5:]
    print(
        f"average error on {len(testing_data)} samples with no training:",
        network.test_network(testing_data)
    )
    network.train_network(training_data)
    print(
        (f"average error on {len(testing_data)} samples after training on "
        f"{len(training_data)} other samples:"),
        network.test_network(testing_data)
    )
