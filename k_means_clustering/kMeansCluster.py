import sys
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns


class bcolors:
    """
    Colors for printing in terminal output
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def plotGraph(points, clusterToPointIndexes, centroids, iterationNo, showGraph):
    """
    Plot graph creates a cluster plotting using matplotlib library.
    This function also stores the plat in the local files system.
    """

    # plotting the graph using the points and
    sns.scatterplot(x=[x[0] for x in points],
                    y=[y[1] for y in points],
                    hue=clusterToPointIndexes,
                    style=clusterToPointIndexes,
                    palette="deep",
                    marker="d",
                    )

    # Plotting the centroids on the graph
    plt.plot([x[0] for x in centroids],
             [y[1] for y in centroids],
             'k+',
             markersize=15,
             )

    plt.title("K-means cluster")

    # Generating filename to store the generated plot graph
    fileName = "kmeans-iteration"+str(iterationNo)+".png"

    # saving file to local file system
    plt.savefig(fileName, facecolor="w", bbox_inches="tight",
                pad_inches=0.3, transparent=True)

    if showGraph == True:
        # Displaying plot graph
        plt.show()

    return


def pointInMacro(points, xLow, yLow, xTop, yTop):
    """
    This function returns the no of points that will present with this x and y values 
    """
    nMacro = 0
    for point in points:

        if (point[0] >= xLow) and (point[0] < xTop) and (point[1] >= yLow) and (point[1] < yTop):
            nMacro += 1
    return nMacro


def generateSeedPoints(points, noOfClusters):
    """
    This function generates a random initial seed points to create the cluster
    output: a list of seed points for each cluster based on no of clusters desired.
    """
    randomSeeds = []

    # getting the x and y points from data points
    xPoints = [x[0] for x in points]
    yPoints = [y[1] for y in points]

    # print("x points length", len(xPoints))

    # calculating min and max of x and y
    xMax = max(xPoints)
    xMin = min(xPoints)
    yMax = max(yPoints)
    yMin = min(yPoints)
    # print("max and mins: ", xMax, xMin, yMax, yMin)

    # Calculating the size of x and y
    xSize = (xMax-xMin)/noOfClusters

    ySize = (yMax-yMin)/noOfClusters

    # Macro blocks initialization
    nMacro = noOfClusters * noOfClusters

    # TODO: need to change this density parameter
    # the current logic produces a high number
    avgDensity = len(points)/nMacro

    # intermediate seed points list
    higherDensityMacros = []
    # print("xSize,ySize,nMacro,avgDensity", xSize, ySize, nMacro, avgDensity)

    # looping to get the highest density points
    for i in range(noOfClusters):
        xLow = xMin+(i * xSize)
        xHigh = xLow+xSize
        xMid = (xLow+xHigh)/2

        # print("xLow,xHigh,xMid", i, xLow, xHigh, xMid)

        for j in range(noOfClusters):
            yLow = yMin+(j*ySize)
            yHigh = yLow+ySize
            yMid = (yLow+yHigh)/2
            # print("j,xLow, yLow, xHigh, yHigh,yMid",
            #       j, xLow, yLow, xHigh, yHigh, yMid)

            nPoint = pointInMacro(points, xLow, yLow, xHigh, yHigh)
            # print("nPoint:", nPoint)

            # adding points to high density seeds
            if nPoint > avgDensity:
                higherDensityMacros.append([xMid, yMid])

    # Quitting the program if high density points added is less than the required number of cluster
    if len(higherDensityMacros) < noOfClusters:
        print(f"{bcolors.FAIL}\n Failed to generate initial seed points as the total high density points ({len(higherDensityMacros)}) generated is less than the desired number of cluster({noOfClusters}) \n{bcolors.ENDC}")
        quit()
    # Randomly adding the initial seed points
    for i in range(noOfClusters):
        seed = random.choice(higherDensityMacros)
        randomSeeds.append(seed)
        higherDensityMacros.remove(seed)

    # TODO: Radius is getting very small as the xSize and ySize is getting too small
    # Assigning 0 so that the radius picked up would be half of the smallest distance between centroid
    radius = 0  # (min([xSize, ySize]))/noOfClusters

    # Calculating the distance.
    for i in range(noOfClusters):
        firstSeed = randomSeeds[i]
        for j in range(noOfClusters):
            if i != j:
                secondSeed = randomSeeds[j]
                distance = math.sqrt(
                    (firstSeed[0]-secondSeed[0])**2+(firstSeed[1]-secondSeed[1])**2)
                # print("distance:", distance)
                if distance < 2*radius or radius == 0:
                    radius = distance/2
                    # print("radius change:", radius)
    # quit()
    return randomSeeds, radius


def getEuclideanDistance(point, centroids):
    """
    This function calculates the euclidean distance between each centroid to the given point
    output: list of distances from point to the centroids
    """

    # Calculating the distance
    distance = [math.sqrt(
        (point[0]-centroid[0])**2+(point[1]-centroid[1])**2) for centroid in centroids]

    return distance


def calculateCentroidThreshold(previousCentroids, currentCentroids):
    """
    This function calculates the centroid shift between old centroid and new centroid
    """

    centroidShift = []
    # Calculating the distance between two centroids.
    for i in range(len(previousCentroids)):

        # Validating if the cluster has any centroid
        # This is required as if a cluster has no points in it.
        # this will result in array out of range error

        if len(currentCentroids) < i+1:
            currentCentroidX = 0
            currentCentroidY = 0
        else:
            currentCentroidX = currentCentroids[i][0]
            currentCentroidY = currentCentroids[i][1]

        # calculating the distance
        centroidShift.append(math.sqrt(
            (previousCentroids[i][0]-currentCentroidX)**2+(previousCentroids[i][1]-currentCentroidY)**2))

    print(f"{bcolors.OKCYAN}\n Centroid shifts: {bcolors.OKBLUE} {centroidShift}{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}\n Maximum centroid shift: {bcolors.OKBLUE} {max(centroidShift)}{bcolors.ENDC}")

    # returning the maximum centroid shift
    return max(centroidShift)


def kMeansClustering(points, desiredClusters, maxIterations, maxAllowedShiftInThreshold):
    """
    This function creates the desired no of clusters on the given data points.
    """

    # Getting the random seed points from the data points
    currentSeeds, radius = generateSeedPoints(points, desiredClusters)

    print(f"{bcolors.OKCYAN}\n Random generated seeds: {bcolors.OKBLUE} {currentSeeds}{bcolors.ENDC}")

    print(f"{bcolors.OKCYAN}\n Random generated radius: {bcolors.OKBLUE} {radius}{bcolors.ENDC}")

    # To store previous centroid points
    previousCentroids = []

    # Counter initialized
    currentIteration = 0

    # flag to indicate weather the cluster has stabilized
    clusterStabilized = False

    # looping until cluster is stabilized
    while clusterStabilized == False and currentIteration < maxIterations:
        print(
            f"{bcolors.BOLD}\n          --------------- Cycle {currentIteration}  ---------------{bcolors.ENDC}")

        print(
            f"{bcolors.OKCYAN}\nCurrent seed points: {bcolors.OKBLUE} {currentSeeds}{bcolors.ENDC}")
        # This variable is important to plot the graph.
        plotGraphClusterIndexes = []

        # stores outliers
        outliers = []

        # initializing the default empty clusters
        clusters = [[] for _ in range(desiredClusters)]
        for point in points:

            # calculating the euclidean distances from point to the centroids
            euclideanDistancesFromPoint = []
            euclideanDistancesFromPoint = getEuclideanDistance(
                point, currentSeeds)

            # Assigning the point to the least distant centroid.
            leastDistance = min(euclideanDistancesFromPoint)

            # checking if the least distant centroid is under the radius.
            if leastDistance > radius:

                # adding the point to outliers if distant is above radius
                outliers.append(point)
                plotGraphClusterIndexes.append("outliers")
            else:
                # Adding the point to the cluster
                clusterIndex = euclideanDistancesFromPoint.index(leastDistance)
                clusters[clusterIndex].append(point)

                # adding point index in the cluster to plot the graph
                plotGraphClusterIndexes.append("cluster"+str(clusterIndex+1))

        # assigning the current seeds to previous centroid
        previousCentroids = currentSeeds

        # removing the current seeds
        currentSeeds = []

        # creating a new centroids for next iteration
        for cluster in clusters:
            if len(cluster) != 0:
                currentSeeds.append(
                    [sum(x[0] for x in cluster)/len(cluster), sum(y[1] for y in cluster)/len(cluster)])
            # else:
            #     print("this cluster has no points:", cluster)
        if len(points) < 100:
            for i in range(len(clusters)):
                print(
                    f"{bcolors.OKCYAN}\n Cluster {i+1} points: {bcolors.OKBLUE} {clusters[i]}{bcolors.ENDC}")
            print(
                f"{bcolors.OKCYAN}\n Outliers: {bcolors.OKBLUE} {outliers}{bcolors.ENDC}")

        else:

            print(
                f"{bcolors.OKCYAN}\n Total data points: {bcolors.OKBLUE} {len(points)}{bcolors.ENDC}")

            for i in range(len(clusters)):
                print(
                    f"{bcolors.OKCYAN}\n Total Cluster {i+1} points: {bcolors.OKBLUE} {len(clusters[i])}{bcolors.ENDC}")
            print(
                f"{bcolors.OKCYAN}\n Total Outliers: {bcolors.OKBLUE} {len(outliers)}{bcolors.ENDC}")

        print(
            f"{bcolors.OKCYAN}\n Previous Centroids: {bcolors.OKBLUE} {previousCentroids}{bcolors.ENDC}")

        print(
            f"{bcolors.OKCYAN}\n New Centroids: {bcolors.OKBLUE} {currentSeeds}{bcolors.ENDC}")
        # Calculating the centroids shift
        centroidShift = calculateCentroidThreshold(
            previousCentroids, currentSeeds)

        print(
            f"{bcolors.OKCYAN}\n New Centroids: {bcolors.OKBLUE} {currentSeeds}{bcolors.ENDC}")

        # validating the cluster stability
        if centroidShift <= maxAllowedShiftInThreshold:
            clusterStabilized = True

        # Plotting the graph
        if clusterStabilized == True:
            # Plotting a graph
            plotGraph(points, plotGraphClusterIndexes,
                      previousCentroids, currentIteration, clusterStabilized)
        print(
            f"{bcolors.OKCYAN}\n Cluster stabilized ? : {bcolors.OKBLUE} {clusterStabilized == True}{bcolors.ENDC}")

        # incrementing the iteration counter.
        currentIteration = currentIteration+1

    return


def readDataPointsFile(pointsDataFileName, errorTxt, retries):
    """
    This function reads the filename from terminal and gets the data points from the file
    output: list of data points
    """

    # end program if max retries reached
    if retries == 0:
        print(
            f"{bcolors.FAIL}\n Maximum retries reached. Ending program\n{bcolors.ENDC}")

        # ending program
        quit()
    try:

        # Printing error message to terminal if any
        if errorTxt != "":
            print(f"{bcolors.FAIL}\n {errorTxt}\n{bcolors.ENDC}")

        if pointsDataFileName == "":
            # Reading the data point filename
            pointsDataFileName = input(
                f"{bcolors.OKBLUE} Please enter the file name where data points are stored: {bcolors.ENDC}")

        # Getting data points from the file
        with open(pointsDataFileName) as pointsFile:
            dataPoints = [list(map(float, point.split(",")))
                          for point in pointsFile]

        return dataPoints

    except:

        # if error in reading in file, retry the operation and reduce the max retries by 1
        readDataPointsFile("", "Invalid file Name", retries - 1)


def getDataFromUser():
    """
    This function asks the user to enter the required data from the terminal
    """
    # Taking the arguments from the command line
    if len(sys.argv) < 4:
        print(f"{bcolors.WARNING}\nIt looks like you have missed to enter the required data or entered an invalid format, when starting this program.\n\n{bcolors.ENDC}")

        # Read the Data points from file.
        points = readDataPointsFile("", "", 3)

        # Read Number of clusters desired
        desiredClusters = int(
            input(f"{bcolors.OKBLUE}\n Enter desired Number of clusters: {bcolors.ENDC}"))

        # Read Maximum iteration permitted for obtaining stability in the cluster
        maxIteration = int(
            input(f"{bcolors.OKBLUE}\n Enter desired maximum iteration: {bcolors.ENDC}"))

        # Read Maximum allowable shift threshold in the centroid in the next cycle
        maxAllowedShiftInThreshold = float(
            input(f"{bcolors.OKBLUE}\n Enter maximum maximum allowable shift of the centroid in next cycle: {bcolors.ENDC}"))

        return points, desiredClusters, maxIteration, maxAllowedShiftInThreshold
    else:

        # Read the Data points from file.
        points = readDataPointsFile(sys.argv[1], "", 2)

        # Read Number of clusters desired
        desiredClusters = int(sys.argv[2])

        # Read Maximum iteration permitted for obtaining stability in the cluster
        maxIteration = int(sys.argv[3])

        # Read Maximum allowable shift threshold in the centroid in the next cycle
        maxAllowedShiftInThreshold = float(sys.argv[4])

        return points, desiredClusters, maxIteration, maxAllowedShiftInThreshold


if __name__ == '__main__':
    points, desiredClusters, maxIteration, maxAllowedShiftInThreshold = getDataFromUser()
    # print(len(points), desiredClusters, maxIteration, maxAllowedShiftInThreshold)

    kMeansClustering(points, desiredClusters, maxIteration,
                     maxAllowedShiftInThreshold)
    quit()
