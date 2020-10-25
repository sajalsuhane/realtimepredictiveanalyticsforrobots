import csv
import random
import math
import operator
import matplotlib.pyplot as plt
from array import *
from matplotlib import colors
from matplotlib import rcParams
import numpy as np



def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def printAccuracy(accuracy):
    if accuracy > 70:
        print('The algorithm does qualifies threshold accuracy value')
    else:
        print('The algorithm does not qualify threshold accuracy value')


def newPredict(predictions, accuracy):
    count_good = 0
    count_bad = 0
    predicted_result = 'Invalid'
    for i in range(len(predictions)):
        if predictions[i] == 'Pass':
            count_good += 1
        else:
            count_bad += 1
    if accuracy > 70:
        if count_good > count_bad:
            predicted_result = 'Pass'
            print('Next result is possibly {}'.format(predicted_result))
        else:
            predicted_result = 'Fail'
            print('Next result is possibly {}'.format(predicted_result))
    else:
        if count_good > count_bad:
            predicted_result = 'Fail'
            print('Next result is possibly {}'.format(predicted_result))
        else:
            predicted_result = 'Pass'
            print('Next result is possibly {}'.format(predicted_result))
    return predicted_result


def plotGraph(testSet, predictions, new_prediction):
    x = len(testSet) - 6
    days_array = []
    new_prediction_plot = []
    fig = plt.figure(figsize=(7, 1))
    fig.tight_layout()
    plt.margins(2, 2)
    for x in range(len(testSet)):
        days_array.append(x)

    for i in range(len(days_array)):

        if testSet[days_array[i]][-1] == 'Pass':
            plt.plot(days_array[i], 1, marker='o', markerfacecolor='black', markersize=12)
            # plt.plot(days_array[i],0.8,linestyle='dashed',markerfacecolor='black',markersize=12)
        else:
            plt.plot(days_array[i], 0, marker='o', markerfacecolor='black', markersize=12)
            # plt.plot(days_array[i],0.2,linestyle='dashed',markerfacecolor='black',markersize=12)

    plt.plot(days_array[0], 999, label='Original', markerfacecolor='red', marker='o')

    if new_prediction == 'Pass':
        days_array.append(len(days_array))
    else:
        days_array.append(len(days_array))

    if new_prediction == 'Pass':
        plt.plot(days_array[len(days_array) - 1], 1, label='Predicted', marker='*', markerfacecolor='blue',
                 markersize=13)
    else:
        plt.plot(days_array[len(days_array) - 1], 0, label='Predicted', marker='*', markerfacecolor='blue',
                 markersize=13)

    for i in range(len(testSet)):
        if predictions[i] == 'Pass':
            new_prediction_plot.append(1)
        else:
            new_prediction_plot.append(0)

    days_array_list = list(days_array)
    new_prediction_plot_list = list(new_prediction_plot)
    # new_prediction = new_prediction_plot_list-1
    # plt.plot(days_array_list,new_prediction)
    for i in range(len(days_array) - 1):
        plt.plot(days_array_list[i], new_prediction_plot_list[i], color='green', linestyle='dashed', linewidth=3,
                 marker='*', markerfacecolor='yellow', markersize=10)
    #             connectpoints(days_array[i],new_prediction_plot[i],i,i+1)
    #   rcParams['xtick.major.pad']='8'
    plt.axis([x, x + 6, 0, 1])
    plt.ylim(0, 3)
    plt.xlim(x - 5, x + 2)
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Result')
    plt.show()


# def connectpoints(days_data,new_prediction_plot,p1,p2):
#        x1,x2 = int(days_data[p1],days_data[p2])
#       y1,y2 = int(new_prediction_plot[p1],new_prediction_plot[p2])
#      plt.plot([x1,x2],[y1,y2],'k-')

def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('iris_data.csv', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    k = 7
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    # train_accuracy = getAccuracy(trainingSet,
    print('Accuracy: ' + repr(accuracy) + '%')
    printAccuracy(accuracy)
    new_prediction = newPredict(predictions, accuracy)
    plotGraph(testSet, predictions, new_prediction)

if __name__ == "__main__":
    main()

main()

