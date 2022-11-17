import math
from random import uniform

with open("atan2.csv", "w+") as data_output:
    for _ in range(0, 100):
        x = uniform(-50, 50)
        y = uniform(-50, 50)
        data_output.write(f"{x},{y},{math.atan2(y,x)}\n")
