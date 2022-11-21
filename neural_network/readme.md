## Neural Network

By Mitch Jacovetty. Written for Python 3.11 with numpy as its only external dependency.

After installing numpy with `pip install numpy --user`, run with `python neural_network.py configuration.json`. Any data not found in configuration.json will be read from the terminal when the program runs.

By default, the network is trained on a simple rounding function that returns 0 for numbers in the range [0, 0.5] and 1 for numbers in the range (0.5, 1]. This function fits well with the output capabilities of the sigmoid function as a simple test case for the algorithm functioning.
