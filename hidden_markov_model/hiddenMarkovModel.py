
import random
import numpy as np

customStates = int(input("Enter number of States : "))
emissionsPerState = int(input("Enter number of Emissions in each state : "))


randomTransmissionValues = [[random.uniform(0, 0.1 / customStates)
                             for i in range(customStates)] for j in range(customStates)]

transitionMatrix = [np.round(item, 2)
                    for item in randomTransmissionValues]

initial = []
for index in range(emissionsPerState):
    initialStateValue = float(
        input("Enter initial state value of "+str(index)+": "))
    initial.append(initialStateValue)

print("\nTransition matrix :", transitionMatrix)
print("\nEntered Initial States: ", initial)

randomEmissionValues = [[random.uniform(0, 0.1 / emissionsPerState)
                         for i in range(emissionsPerState)] for j in range(emissionsPerState)]

emissionsMatrix = [np.round(item, 2) for item in randomEmissionValues]
print("\nEmissions Matrix:", emissionsMatrix)


Emission = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 23, 'X': 24, 'Y': 25,
            'Z': 26, 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 23, 'x': 24, 'y': 25, 'z': 26}


PathEmission = []

numberOfPathEmission = int(input("Enter number of Path Emission : "))

for i in range(0, numberOfPathEmission):
    element = input("enter element"+str(i)+": ")
    PathEmission.append(element)

print("\n Path Emission:", PathEmission)


pathEmissionInIndexes = [Emission[pe] for pe in PathEmission]
print("\n path Emission InIndexes:", pathEmissionInIndexes)

test_list = [1, 2, 3, 4]
test_tup = (1, 2, 3, 4)

cartesianProducts = [(a, b, c, d)
                     for a in test_tup for b in test_tup for c in test_tup for d in test_tup]

print("The Cartesian Product is : " + str(cartesianProducts))

allProbabilities = []

for i in cartesianProducts:
    stateCounter = 0
    probability = 1
    previousTransitionState = 0
    for x in i:

        if (stateCounter == 0):
            probability = probability*initial[x-1] * \
                emissionsMatrix[x-1][pathEmissionInIndexes[stateCounter]]
            stateCounter += 1
            previousTransitionState = x
        else:
            probability = probability * (transitionMatrix[previousTransitionState-1][x-1]*emissionsMatrix[x-1]
                                         [pathEmissionInIndexes[stateCounter]])
            previousTransitionState = x
            stateCounter += 1
    probability = round(probability, 10)
    allProbabilities.append(probability)
    print(f"Probability of state {i} : {probability} \n")

print(
    f"Best possible path: {cartesianProducts[allProbabilities.index(max(allProbabilities))]}  \n")
