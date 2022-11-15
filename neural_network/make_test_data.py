import math
from random import random

with open("atan2.csv", "w+") as data_output:
    for _ in range(0, 100):
        x = random()*100
        y = random()*100-50
        data_output.write(f"{x},{y},{math.atan2(y,x)}\n")
