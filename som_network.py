import csv
import math
import os
import random
import sys
from hexalattice.hexalattice import *


NEURON_SIZE = 61
LEARNING_RATE = 0.5
MAX_EPOCHS = 10
NEIGHBOR1_MODIFIER = 0.3
NEIGHBOR2_MODIFIER = 0.2
NEIGHBOR3_MODIFIER = 0.1
DECAY = 0.1

currentEpoch = 1
cities = []
neurons = []
rowLength = 0


class Neuron:
    def __init__(self):
        self.connectedCities = 0
        self.weights = [random.random() for _ in range(0, rowLength)]
        self.connectedCitiesNames = []


# Calculates distance between a city vector and a given vector of weights
def calcDistance(arrCity, arrWeights):
    distance = 0
    for (data, weight) in zip(arrCity, arrWeights):
        distance += (data - weight) ** 2
    ################################return math.e ** -sum
    return distance ** -2


# Updates the weights of a neuron in the specified index using a received distance vector.
# Optional parameter 'factor', can be  specified for different weight assignments of different range neighbors
def updateWeights(index, city, distance, minDistance, factor=1):
    learningRate = LEARNING_RATE * (math.e ** -(currentEpoch - 1))
    if distance != minDistance:
        neighborDistance = math.e ** (-((distance ** 2) / (2 * (minDistance ** 2))))
        ############################ neighborDistance = (-distance / minDistance)
    else:
        neighborDistance = 1

    #weightChangeFactor = neighborDistance * learningRate
    weightChangeFactor = neighborDistance * (learningRate * (DECAY ** currentEpoch))

    for i in range(0, len(neurons[index].weights)):
        neurons[index].weights[i] += (weightChangeFactor * (city[i + 1] - neurons[index].weights[i])) * factor


# Performs a single epoch iteration, updates weights of neuron closest to any given city, and its closest neighbors
def calculate():
    # re-initialize all neurons, since fresh data will be gathered
    for neuron in neurons:
        neuron.connectedCities = 0
        neuron.connectedCitiesNames = []
    # iterate over all cities
    for city in cities:
        distanceVector = [] # Stores the euclidian distances of each neuron from the city.
        for neuron in neurons:
            # add distance of each and every neuron to the distance vector
            distanceVector.append(calcDistance(city[1:len(city)], neuron.weights))
        # store minimal distance and proximity distance cutoffs
        minDistance = min(distanceVector)
        neighborhood1 = minDistance * 1.1
        neighborhood2 = minDistance * 1.2
        neighborhood3 = minDistance * 1.3

        # update minimal distance neuron
        neurons[distanceVector.index(minDistance)].connectedCitiesNames.append(city[0])
        neurons[distanceVector.index(minDistance)].connectedCities += 1
        updateWeights(distanceVector.index(minDistance), city, minDistance, minDistance)

        # update weight of proximal neurons
        for i in range(0, len(distanceVector)):
            if minDistance < distanceVector[i] <= neighborhood1:
                updateWeights(i, city, distanceVector[i], minDistance, NEIGHBOR1_MODIFIER)
            elif neighborhood1 < distanceVector[i] <= neighborhood2:
                updateWeights(i, city, distanceVector[i], minDistance, NEIGHBOR2_MODIFIER)
            elif neighborhood2 < distanceVector[i] <= neighborhood3:
                updateWeights(i, city, distanceVector[i], minDistance, NEIGHBOR3_MODIFIER)



# converts each city's data into a fraction of the total amount of votes
def toFloat():
    for item in cities:
        sum = 0
        for i in range(1, len(item)):
            item[i] = int(item[i])
            sum += item[i]
        # Convert data to percentage
        for i in range(1, len(item)):
            item[i] = item[i] / sum


# Read data from file
def readFile(fileName):
    with open(fileName, newline='') as file:
        csvFile = csv.reader(file, delimiter=',', quotechar='|')
        global rowLength
        global cities
        for row in csvFile:
            rowLength = len(row) - 1
            cities.append(row)
        cities.pop(0)


# finds neuron with most connected cities
def getMaxCityCount():
    maxCities = 0
    for neuron in neurons:
        if neuron.connectedCities > maxCities:
            maxCities = neuron.connectedCities
    return maxCities


# Write results to csv file
def writeCSV():
    with open("results.csv", "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        i = 0
        writer.writerow(["neuronNum", "city"])
        for neuron in neurons:
            for city in neuron.connectedCitiesNames:
                writer.writerow([i, city])
            i += 1


# interpolates color based on received fraction t
def interpolate(colorMax, colorMin, t):
    return np.array([a + (b - a) * t for a, b in zip(colorMax, colorMin)])


def main():
    fileName = sys.argv[1]
    # error check
    if not os.path.exists(fileName):
        print(f"Error: {fileName} File does not exist")
        exit(-1)
    # read .csv file, initialize data
    readFile(fileName)
    # Convert read strings into floats
    toFloat()
    # create a neuron array
    for i in range(0, NEURON_SIZE):
        neuron = Neuron()
        neurons.append(neuron)

    # perform MAX_EPOCHS of epochs
    global currentEpoch
    for currentEpoch in range(1, MAX_EPOCHS):
        calculate()

    # Drawing hex grid
    hex_centers, _ = create_hex_grid(n=100, crop_circ=4, do_plot=False)
    x_hex_coords = hex_centers[:, 0]
    y_hex_coords = hex_centers[:, 1]

    colors = np.zeros([61, 3])              # Create empty array of colors
    colorMax = np.array([1, 1, 1])    # Color representing most connected neuron
    colorMin = np.array([0, 0, 0])          # Color representing the least connected neuron
    maxCities = getMaxCityCount()
    i = 0
    # Assign color to each neuron based on connected cities
    for neuron in neurons:
        colors[i] = interpolate(colorMax, colorMin, neuron.connectedCities / maxCities)
        i += 1

    plot_single_lattice_custom_colors(x_hex_coords, y_hex_coords,
                                      face_color=colors,
                                      edge_color=np.zeros([61, 3]),
                                      min_diam=0.9,
                                      plotting_gap=0.05,
                                      rotate_deg=0)

    # save results
    plt.savefig("results.png")
    print("Done!\n")
    print("Grid saved to results.png")
    print("Results written to results.csv")
    writeCSV()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path_to_csv_file>")
    else:
        main()
