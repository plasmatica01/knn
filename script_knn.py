# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:49:43 2019

@author: L0321168

"""
from random import seed
from random import randrange
from math import sqrt
import pandas as pd
import numpy as np

class KNN_classifier ():
    
#First think, iniciate the class    
    def __init__(self):  
        pass 
    
 # Find the min and max values for each column     
    def dataset_minmax(self, dataset):
    	minmax = list()
    	for i in range(len(dataset[0])):
    		col_values = [row[i] for row in dataset]
    		value_min = min(col_values)
    		value_max = max(col_values)
    		minmax.append([value_min, value_max])
    	return minmax

# Rescale dataset columns to the range 0-1
    def normalize_dataset(self, dataset):
        minmax = self.dataset_minmax(dataset)
        for row in dataset:
            for i in range (len(row)):
                row [i] = (row [i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])            
  
# Split a dataset into k folds, to have train and test data
    def cross_validation_split(self, dataset, n_folds):
    	dataset_split = list()
    	dataset_copy = list(dataset)
    	fold_size = int(len(dataset) / n_folds)
    	for _ in range(n_folds):
    		fold = list()
    		while len(fold) < fold_size:
    			index = randrange(len(dataset_copy))
    			fold.append(dataset_copy.pop(index))
    		dataset_split.append(fold)
    	return dataset_split
    
# Calculate accuracy percentage, to have one form to evalaute the algorithm
    def accuracy_metric(self, actual, predicted):
    	correct = 0
    	for i in range(len(actual)):
    		if actual[i] == predicted[i]:
    			correct += 1
    	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
    	folds = self.cross_validation_split(dataset, n_folds)
    	scores = list()
    	for fold in folds:
    		train_set = list(folds)
    		train_set = sum(train_set, [])
    		test_set = list()
    		for row in fold:
    			row_copy = list(row)
    			test_set.append(row_copy)
    			row_copy[-1] = None
    		predicted = algorithm(train_set, test_set, *args)
    		actual = [row[-1] for row in fold]
    		accuracy = self.accuracy_metric(actual, predicted)
    		scores.append(accuracy)
    	return scores
 
# Calculate the Euclidean distance between two vectors
    def euclidean_distance(self, row1, row2):
    	distance = 0.0
    	for i in range(len(row1)-1):
    		distance += (row1[i] - row2[i])**2
    	return sqrt(distance)
 
# Locate the most similar neighbors, k is the number of neighbours I want
    def get_neighbors(self, train, test_row, k):
    	distances = list()
    	for train_row in train:
    		dist = self.euclidean_distance(test_row, train_row)
    		distances.append((train_row, dist))
    	distances.sort(key=lambda tup: tup[1])
    	neighbors = list()
    	for i in range(k):
    		neighbors.append(distances[i][0])
    	return neighbors
 
# Make a prediction with neighbors
    def predict (self, train, test_row, k):
    	neighbors = self.get_neighbors(train, test_row, k)
    	output_values = [row[-1] for row in neighbors]
    	prediction = max(set(output_values), key=output_values.count)
    	return prediction
     
# kNN Algorithm
    def k_nearest_neighbors(self, train, test, k):
    	predictions = list()
    	for row in test:
    		output = self.predict (train, row, k)
    		predictions.append(output)
    	return(predictions)
 
   
#Example of usage
#Set a seed, create a dataset and convert it to array with the binary target 
np.random.seed(0)
df = pd.DataFrame(np.random.randint(-100,100,size=(800, 2)).astype("float"), columns=['feature_1','feature_2'])
df['target'] = np.random.randint(2,size=800)
z = df.copy()
data=z.values 

# Initiate the classifier and normalize the data
knn=KNN_classifier ()
knn.normalize_dataset(data)
#If I would like to see how it works, I define a new record
test_row = [0.15, 0.32 ,0]
#I could have the 3 nearest neighbors which are the following
knn.get_neighbors(data, test_row, 3)
#If I would like to see which class the algorithm will predict for this new point
prediction=knn.predict (data, test_row, 3)
print('Expected %d, Got %d.' % (test_row[-1], prediction))
#The prediction is 0, that is logical because the class of the 3 nearest neighbors is 0, so I didn't expect something different
# I could do it with more records, example:
prediction= knn.k_nearest_neighbors(data, data[6:10], 3)
#If I want to use cross validation split to fit my algorithm and then see how it performs
#I could use the accurary measure
#If I have 4 folds and the data has 800 records, then I´d have folds of 200 records
#I´m considering in this case, just 3 neighbors, but I could change it
seed(1)
n_folds = 4
k = 3
scores = knn.evaluate_algorithm(data, knn.k_nearest_neighbors, n_folds, k)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores)))) 
#I have 74%, 74.5%, 80% and 76% of accuracy in each fold
#I could use the mean to have a measure of the performance=76.125% 
#If I try with 10 neighbors the mean accuracy decreases to 60% 
seed(1)
n_folds = 4
k = 10
scores = knn.evaluate_algorithm(data, knn.k_nearest_neighbors, n_folds, k)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores)))) 

#If I want the probability instead of the class I could use the average instead of the max (because the classes are 0 and 1)

    


    

