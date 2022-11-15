import sys

class Line:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    
    def get_y(self, x: float) -> float:
        return self.beta*x+self.alpha
    
    @classmethod
    def regression_from_points(cls, points: list[tuple[float, float]]) -> "Line":
        x_mean = sum(p[0] for p in points)/len(points)
        y_mean = sum(p[1] for p in points)/len(points)
        x_variance = (
            sum((p[0]-x_mean) ** 2 for p in points) /
            (len(points) - 1)
        )
        covariance = (
            sum((p[0] - x_mean) * (p[1] - y_mean) for p in points) /
            (len(points) - 1)
        )
        beta = covariance/x_variance
        alpha = y_mean-beta*x_mean
        return cls(alpha, beta)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("filename argument for data required")
    else:
        with open(sys.argv[1]) as datafile:
            points = []
            for point in datafile:
                points.append(tuple(map(float, point.split(","))))
            line = Line.regression_from_points(points)
            user_input = ""
            while True:
                user_input = input("Enter an x to receive the estimated y, or enter \"stop\" to stop: ")
                if user_input == "stop":
                    break
                else:
                    try:
                        x = float(user_input)
                        print("y: ", line.get_y(x))
                    except:
                        print("invalid input :(")
