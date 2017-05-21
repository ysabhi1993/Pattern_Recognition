


# knn for chess data with plot
# kNN implemented algorithm , change the datasets to be run upon in the mainfunction

import csv
import random
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)

        dataset = list(lines)
        print dataset


        stringlist = {}
        count =0
        for x in range(len(dataset)):
            for y in range(4):
                sum1 =0
                if type(dataset[x][y]) is str:
                    if(dataset[x][y] in stringlist ):
                        dataset[x][y] = stringlist[dataset[x][y]]
                    else:
                        count = count +1
                        dataset[x][y] = stringlist[dataset[x][y]]= count

                else:
                    dataset[x][y] = float(dataset[x][y].strip())

            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length): # to calculate the eucledean distance between two points.
    distance = 0
    for x in range(length):
        try:
            distance += pow((instance1[x] - instance2[x]), 2)
            return math.sqrt(distance)
        except TypeError:
            print instance2[x] + " and  " + instance1[x]


def getNeighbors(trainingSet, testInstance, k): # function to get K nearest neighoburs of a given test data point
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length) # to compute the eucledian distance
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1)) # sorting the distance in ascending order.
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0]) # choosing the K nereast points from all the distances
    return neighbors


def getResponse(neighbors): # to count the dominant value among the neighbours and assign that value to the test data point
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:    # couning the number of attributes to determine the dominant value
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    start_time = time.time()
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67

    df = pd.read_csv('kr-vs-kp.data.csv');

    # df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'final']
    a = df.ix[:, 0:4].values
    b = df.ix[:, 4].values
    # X_std = StandardScaler().fit_transform(a);
    # dfcolor = pd.DataFrame([[0, 'red'], [1, 'blue']],columns=['a', 'c'])







    # important functions to get reduced PCA-------------------------------------------------------------------

    pca = PCA(n_components=2)  # 2 PCA components
    pca.fit(df)
    df1 = pd.DataFrame(pca.transform(df))
    # df1.to_csv('chess_pca_20.csv', sep=',')
    # y = pca.explained_variance_
    # print y
    # x = np.arange(len(y)) + 1
    # plt.plot(x, y)
    # plt.show()





    Y_pca = pd.DataFrame(pca.fit_transform(df1),columns=['x','y'])
    # # mergeddf = pd.merge(df, dfcolor)

    # # Then we do the graph
    # plt.scatter(Y_pca[:, 0], Y_pca[:, 1],)
    plt.scatter(Y_pca['x'],Y_pca['y'],c=['r','b'])
    plt.show()





    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    # fig = plt.figure(1, figsize=(4, 3))
    # plt.clf()
    # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    #
    # plt.cla()
    # pca = PCA(n_components=3)
    #
    # pca.fit(X)
    # X = pca.transform(X)
    # for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    #     ax.text3D(X[y == label, 0].mean(),
    #               X[y == label, 1].mean() + 1.5,
    #               X[y == label, 2].mean(), name,
    #               horizontalalignment='center',
    #               bbox=dict(alpha=.7, edgecolor='w', facecolor='w'))
    # # Reorder the labels to have colors matching the cluster results
    # y = np.choose(y, [1, 2, 0]).astype(np.float)
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,cmap=plt.cm.spectral)
    #
    # ax.w_xaxis.set_ticklabels([])
    # ax.w_yaxis.set_ticklabels([])
    # ax.w_zaxis.set_ticklabels([])


    loadDataset('chess_pca_20.csv', split, trainingSet, testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    # generate predictions
    predictions = []
    d = []
    e = []


    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)


        # d.append(testSet[x][1])
        # e.append(testSet[x][2])
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

        plt.show() #plot show for the 3d graph of iris data

    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    # plt.scatter(d, e, c=['r', 'b', 'g'])
    # plt.show()


    elbow = {}
    elbow[3] = 100-98.05
    elbow[1] = 100-95.36
    elbow[2] = 100-95.98
    elbow[4] = 100-95.96
    elbow[5] = 100-95.85

    elbow[3] = 100 - 98.05
    elbow[1] = 100 - 95.36
    elbow[2] = 100 - 95.98
    elbow[4] = 100 - 95.96
    elbow[5] = 100 - 95.85

    a = []
    b = []
    for key in elbow:
        a.append(key)
        b.append(elbow[key])
    print a, b
    plt.plot(a,b) # elbow plot for knn
    plt.ylabel('percentage of ERROR')
    plt.xlabel('K value')
    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))

main()