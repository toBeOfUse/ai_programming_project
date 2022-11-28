import random
import numpy as np
import itertools

customStates = int(input("Enter number of States : ")
                   )  # input number of states
# input number of emissions
emissionsPerState = int(input("Enter number of Emissions in each state : "))

# creating random values for the transmission matrix
randomTransmissionValues = [[random.uniform(0, 1 / customStates)
                             for i in range(customStates)] for j in range(customStates)]
# rounding the transmission values to 2
transitionMatrix = [np.round(item, 2)
                    for item in randomTransmissionValues]
print(transitionMatrix)
# input the initial probabilities
initial = []
for index in range(customStates):
    initialStateValue = float(
        input("Enter initial probability of "+str(index+1)+": "))
    initial.append(initialStateValue)

# creating random values for the emission matrix
randomEmissionValues = [[random.uniform(0, 1 / emissionsPerState)
                         for i in range(emissionsPerState)] for j in range(customStates)]

emissionsMatrix = [np.round(item, 2) for item in randomEmissionValues]

print(emissionsMatrix)
Emission = {'A': 0, 'B': 1, 'C': 2, 'D': 3,
            'E': 4, 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}


PathEmission = []
# input number of path emission
numberOfPathEmission = int(input("Enter number of Path Emission : "))
# input the sequence
for i in range(0, numberOfPathEmission):
    element = input("enter element"+str(i)+": ")
    PathEmission.append(element)

print("\nTransition matrix :", transitionMatrix)
print("\nEntered Initial States: ", initial)
print("\nEmissions Matrix:", emissionsMatrix)
print("\n Path Emission:", PathEmission)

# to get emission sequence in integer
pathEmissionInIndexes = [Emission[pe] for pe in PathEmission]
print("\n path Emission InIndexes:", pathEmissionInIndexes)


test_tup = [i for i in range(1, customStates+1)]

cartesianProducts = list(itertools.product(
    test_tup, repeat=numberOfPathEmission))

# printing the possible sequences
print("The Possible Sequences : ", str(cartesianProducts))

# calculating the probability for the cartesian product
allProbabilities = []

for i in cartesianProducts:
    stateCounter = 0
    probability = 1
    previousTransitionState = 0
    for x in i:

        if (stateCounter == 0):
            probability = probability * \
                initial[x-1] * emissionsMatrix[x -
                                               1][pathEmissionInIndexes[stateCounter]]
            stateCounter += 1
            previousTransitionState = x
        else:
            probability = probability * \
                (transitionMatrix[previousTransitionState-1][x-1] *
                 emissionsMatrix[x-1][pathEmissionInIndexes[stateCounter]])
            previousTransitionState = x
            stateCounter += 1
    probability = round(probability, 10)
    allProbabilities.append(probability)
    print(f"Probability of state {i} : {float(probability)} \n")
# printing the best possible path
print(
    f"Best possible path: {cartesianProducts[allProbabilities.index(max(allProbabilities))]}  \n")
