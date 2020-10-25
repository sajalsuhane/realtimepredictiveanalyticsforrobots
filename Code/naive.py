import csv
import random
import math
import matplotlib.pyplot as plt


# from matplotlib import colors

# function for loading the dataset from file in float data type
def loadCsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


# function for splitting the dataset in training and test data
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


# separating the results with dataset
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


# function for calculating mean
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# function for calculating standard deviation
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


# summarizing the dataset based on mean and standard deviation
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


# summarizing the data by class value, if it is not already separated
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


# function for calulating probablity
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# calculating probablities for each class with help of mean and standard deviation
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


# predicting values according to the class probablities and assigning the best probablity value as bestLabel
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


# predictions for the test data
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


# getting accuracy of the algorithm based on test data
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        print('Prediction={0} and Original Result={1}'.format(predictions[i], testSet[i][-1]))
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


# test of algorithm through predefined threshold
def testAccuracy(accuracy):
    if accuracy > 70:
        print('PASS')
    else:
        print('FAIL')


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

        if testSet[days_array[i]][-1] == 1.0:
            plt.plot(days_array[i], 1, marker='o', markerfacecolor='black', markersize=12)
            # plt.plot(days_array[i],0.8,linestyle='dashed',markerfacecolor='black',markersize=12)
        else:
            plt.plot(days_array[i], 0, marker='o', markerfacecolor='black', markersize=12)
            # plt.plot(days_array[i],0.2,linestyle='dashed',markerfacecolor='black',markersize=12)

    plt.plot(days_array[0], 999, label='Original', markerfacecolor='red', marker='o')

    if new_prediction == 1.0:
        days_array.append(len(days_array))
    else:
        days_array.append(len(days_array))

    if new_prediction == 1.0:
        plt.plot(days_array[len(days_array) - 1], 1, label='Predicted', marker='*', markerfacecolor='blue',
                 markersize=13)
    else:
        plt.plot(days_array[len(days_array) - 1], 0, label='Predicted', marker='*', markerfacecolor='blue',
                 markersize=13)

    for i in range(len(testSet)):
        if predictions[i] == 1.0:
            new_prediction_plot.append(1)
        else:
            new_prediction_plot.append(0)

    days_array_list = list(days_array)
    new_prediction_plot_list = list(new_prediction_plot)
    for i in range(len(days_array) - 1):
        plt.plot(days_array_list[i], new_prediction_plot_list[i], color='green', linestyle='dashed', linewidth=3,marker='*', markerfacecolor='yellow', markersize=10)
    #           connectpoints(days_array_list[i],new_prediction_plot_list[i],i,i+1)
    #   rcParams['xtick.major.pad']='8'
    plt.axis([x, x + 6, 0, 1])
    plt.ylim(0, 3)
    plt.xlim(x - 5, x + 2)
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Result')
    plt.show()


# predicting the next pass or fail of the product based on dataset
def newPredict(predictions, accuracy):
    count_no = 0
    count_yes = 0
    predicted_result = 'Invalid'
    for i in range(len(predictions)):
        if predictions[i] == 0.0:
            count_no += 1
        else:
            count_yes += 1
    if accuracy > 70:
        if count_yes > count_no:
            predicted_result = 1.0
            print('Next prediction will be {}'.format(predicted_result))
        else:
            predicted_result = 0.0
            print('Next prediction will be {}'.format(predicted_result))
    else:
        if count_yes > count_no:
            predicted_result = 0.0
            print('Next prediction will be {}'.format(predicted_result))
        else:
            predicted_result = 1.0
            print('Next prediction will be {}'.format(predicted_result))
    return predicted_result


def main():
    filename = 'iris_data_naive.csv'
    splitRatio = 0.90
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test= {2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Algorithm Accuracy:{}%'.format(accuracy))
    testAccuracy(accuracy)
    new_prediction = newPredict(predictions, accuracy)
    plotGraph(testSet, predictions, new_prediction)

if __name__ == "__main__":
    main()

main()
