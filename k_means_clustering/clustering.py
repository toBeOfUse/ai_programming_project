import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("filename argument for points required")
    else:
        with open(sys.argv[1]) as datafile:
            points = [tuple(map(float, point.split(","))) for point in datafile]
            print(points)
