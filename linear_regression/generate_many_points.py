from random import uniform
x = [uniform(-100, 100) for _ in range(100)]
with open("points.csv", "w+") as points_file:
    for x_ in x:
        points_file.write(f"{x_},{(2*x_+5)+uniform(-50,50)}\n")