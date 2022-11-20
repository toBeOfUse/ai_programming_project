import math
from random import uniform

def atan2():
    with open("atan2.csv", "w+") as data_output:
        for _ in range(0, 10000):
            x = uniform(-10, 10)
            y = uniform(-10, 10)
            data_output.write(f"{x},{y},{math.atan2(y,x)}\n")

if __name__ == "__main__":
    atan2()
