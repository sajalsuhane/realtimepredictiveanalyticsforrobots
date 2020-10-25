# Logistic Regression on Diabetes Dataset
from random import seed
from random import randrange
from csv import reader
from math import exp
import csv
import random
import math
import matplotlib.pyplot as plt


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        #                print('Prediction={0} and Original Result={1}'.format(predicted[i], actual[i][-1]))
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))


def testAccuracy(accuracy):
    if accuracy > 70:
        print('Algorithm PASS with accuracy {}%'.format(accuracy))
    else:
        print('Algorithm FAIL with accuracy {}%'.format(accuracy))


def plotGraph(test_set, predictions, new_prediction):
    x = len(test_set) - 6
    days_array = []
    new_prediction_plot = []
    fig = plt.figure(figsize=(7, 1))
    fig.tight_layout()
    plt.margins(2, 2)
    for x in range(len(test_set)):
        days_array.append(x)

    for i in range(len(days_array)):

        if test_set[days_array[i]][-1] == 1.0:
            plt.plot(days_array[i], 1, marker='o', markerfacecolor='black', markersize=12)
            # plt.plot(days_array[i],0.8,linestyle='dashed',markerfacecolor='black',markersize=12)
        else:
            plt.plot(days_array[i], 0, marker='o', markerfacecolor='black', markersize=12)
            # plt.plot(days_array[i],0.2,linestyle='dashed',markerfacecolor='black',markersize=12)

    plt.plot(days_array[0], 9999, label='Original', markerfacecolor='red', marker='o')

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

    if predictions == 1.0:
        new_prediction_plot.append(1)
    else:
        new_prediction_plot.append(0)

    days_array_list = list(days_array)
    new_prediction_plot_list = list(new_prediction_plot)
    #    for i in range(len(days_array)-1):
    #           plt.plot(days_array_list[i],new_prediction_plot_list[i],color='green',linestyle='dashed',linewidth=3,marker='*',markerfacecolor='yellow',markersize=10)
    #             connectpoints(days_array[i],new_prediction_plot[i],i,i+1)
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


# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
    return coef


# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coef)
        yhat = round(yhat)
        #		print('Expected= {0}, Predicted= {1}'.format((row[-1], yhat)))
        predictions.append(yhat)
    return (predictions)


# Test the logistic regression algorithm on the dataset
seed(1)

# load and prepare data
# filename = 'pima-indians-diabetes.csv'
filename = 'final_dataset_12.csv'
dataset = load_csv(filename)
splitRatio = 0.90
trainingSet, testSet = splitDataset(dataset, splitRatio)
print('Split {0} rows into train={1} and test= {2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# normalize
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.1
n_epoch = 100
predictions = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
accuracy = (sum(predictions) / float(len(predictions)))
testAccuracy(accuracy)
new_prediction = logistic_regression(trainingSet, testSet, l_rate, n_epoch)
plotGraph(testSet, predictions, new_prediction)

