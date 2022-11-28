import math
from random import uniform

def atan2():
    with open("atan2.csv", "w+") as data_output:
        for _ in range(10000):
            x = uniform(-10, 10)
            y = uniform(-10, 10)
            data_output.write(f"{x},{y},{math.atan2(y,x)}\n")

def cos():
    with open("cos.csv", "w+") as data_output:
        for _ in range(10000):
            x = uniform(-100, 100)
            data_output.write(f"{x},{math.cos(x)}\n")

def sin():
    with open("sin.csv", "w+") as data_output:
        for _ in range(10000):
            x = uniform(-10, 10)
            data_output.write(f"{x},{math.sin(x)}\n")

def round():
    with open("round.csv", "w+") as data_output:
        for _ in range(10000):
            x = uniform(0, 1)
            data_output.write(f"{x},{1.0 if x>0.5 else 0.0}\n")

def quandrants():
    with open("quadrants.csv", "w+") as data_output:
        for _ in range(10000):
            x = uniform(-1, 1)
            y = uniform(-1, 1)
            q1 = 1 if x>=0 and y>=0 else 0
            q2 = 1 if x<0 and y>=0 else 0
            q3 = 1 if x<0 and y<0 else 0
            q4 = 1 if x>=0 and y<0 else 0
            data_output.write(",".join(map(str,[x,y,q1,q2,q3,q4]))+"\n")

if __name__ == "__main__":
    quandrants()
