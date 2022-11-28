import random
import numpy as np

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
Emission = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 23, 'X': 24, 'Y': 25,
            'Z': 26, 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 23, 'x': 24, 'y': 25, 'z': 26}


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


test_tup = [1, 2, 3, 4]

cartesianProducts = [[a, b, c, d]
                     for a in test_tup for b in test_tup for c in test_tup for d in test_tup]

# printing the possible sequences
print("The Possible Sequences : " + str(cartesianProducts))

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
