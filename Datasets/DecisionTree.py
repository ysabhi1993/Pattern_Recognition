# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import math
import operator
import numpy as np
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd

actual_total = []
predicted_total = []
another_total = []

# Load a CSV file
def load_csv(filename):
	file  = open(filename, "r", encoding='ascii')
	lines = reader(file, delimiter = ',')
	dataset = list(lines)
	return dataset

hashMap = {}

# Convert string column to float
def str_column_to_float(dataset, column):
	i = 0
	for row in dataset:
		if type(row[column]) is str:
			if row[column] in hashMap:
				row[column] = hashMap[row[column]]
			else:
				i = i + 1
				hashMap[row[column]] = i 
				row[column] = i
		else:
			row[column] = float((row[column].strip()))

def get_column(dataset):
	for i in range(len(dataset)):
		another_total.append(dataset[i][-1])

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
		actual_total.append(actual)
		predicted_total.append(predicted)
		scores.append(accuracy)
	return scores

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
	gini = 0.0
	for class_value in class_values:
		for group in groups:
			size = len(group)
			if size == 0:
				continue
			proportion = [row[-1] for row in group].count(class_value) / float(size)
			gini += (proportion * (1.0 - proportion))
	return gini

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)

# Test CART on Bank Note dataset
seed(1)
# load and prepare data
filename = 'HTRU_2.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
n_folds = 5
max_depth = 50
min_size = 10
scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
pred_total = []
for i in range(n_folds):
	for j in predicted_total[i]:
		pred_total.append(j)
get_column(dataset)

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

#df = pd.read_csv(filename)
#important functions to get reduced PCA-------------------------------------------------------------------

pca = PCA(n_components=2)  # 2 PCA components
pca.fit(dataset)
df1 = pd.DataFrame(pca.transform(dataset))
df1.to_csv('After_PCA.csv', sep=',')
y = pca.explained_variance_
print (y)
x = np.arange(len(y)) + 1
plt.plot(x, y)
plt.show()
'''
Y_pca = pd.DataFrame(pca.fit_transform(df),columns=['x','y'])
mergeddf = pd.merge(df, dfcolor)
#Then we do the graph

plt.scatter(Y_pca[:, 0], Y_pca[:, 1],color=mergeddf['Color'])
plt.scatter(Y_pca['x'],Y_pca['y'],c=['r','b'])
plt.show()
'''
#df = pd.read_csv()
#dataset = load_csv('After_PCA.csv')
#scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
X_reduced = PCA(n_components=3).fit_transform(dataset)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=['Y', 'R'], cmap=plt.cm.Paired)
#Axes3D.scatter(X_reduced[:,0], X_reduced[:,1], X_reduced[:,2], c=['Y', 'R', 'B'], cmap=plt.cm.Paired)
plt.show()

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
