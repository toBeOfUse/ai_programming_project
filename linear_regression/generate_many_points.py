from random import uniform
import sys
import re

if len(sys.argv) == 2:
    try:
        equation_regex = r"^(\d+(?:\.\d+)?)x\+(\d+(?:\.\d+)?)$"
        if (eq_comps := re.match(equation_regex, sys.argv[1])) is not None:
            beta = float(eq_comps[1])
            alpha = float(eq_comps[2])
            print(f"read {beta}x+{alpha} from the command line")
    except Exception as e:
        print("could not read linear equation from command line")
        print(e)
        exit()
else:
    beta = 2
    alpha = 5
x = [uniform(-100, 100) for _ in range(100)]
with open("points.csv", "w+") as points_file:
    for x_ in x:
        points_file.write(f"{x_},{(beta*x_+alpha)+uniform(-10,10)}\n")