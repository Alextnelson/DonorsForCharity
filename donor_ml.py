#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 00:13:39 2017

@author: alexandernelson
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
print(data.head(n=10))
print(type(data))
print(type(data['income'][0]))

#Total number of records
n_records = len(data.index)
print(n_records)
print(type(data['income']))
income_counts = data['income'].value_counts()
print(income_counts)

# Number of records where individual's income is more than $50,000
n_greater_50k = income_counts['>50K']

# Number of records where individual's income is at most $50,000
n_at_most_50k = income_counts['<=50K']

# Percentage of individuals whose income is more than $50,000
greater_percent = n_greater_50k / n_records * 100

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)
print(type(income_raw))

# Visualize skewed continuous features of original data
vs.distribution(data)

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_raw, transformed = True)

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# Show an example of a record with scaling applied
display(features_raw.head(n = 1))

# One-hot encode the 'features_raw' data using pandas.get_dummies()
features = pd.get_dummies(features_raw)

# Encode the 'income_raw' data to numerical values
income = [0 if value == '<=50K' else 1 for value in income_raw]
print(income[1:10])
print(type(income))



# Print the number of features after one-hot encoding
encoded = list(features.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Print the encoded feature names
print(encoded)

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

print(X_train.head(n=2))
print(y_train)

from sklearn.metrics import accuracy_score

test_sample_size = X_test.shape[0]
all_greater_50k = [1]*(test_sample_size)
print(len(all_greater_50k))

# Calculate accuracy
accuracy = float(accuracy_score(all_greater_50k, y_test))
print("Naive Predictor has accuracy: ", accuracy)
print(type(accuracy))

# Calculate precision with true positives/(true positives + false positives)
# Calculate recall with true positives/(true positives + false negatives)
# Keep in mind that precision is accuracy restated & recall is 1 since the Naive model identifies all as income >50K
true_positives = accuracy * test_sample_size
print(true_positives)
false_positives = test_sample_size - true_positives
precision = true_positives/(true_positives + false_positives)
print(precision)
false_negatives = 0
recall = true_positives/(true_positives + false_negatives)
print(recall)
print(type(recall))

# Calculate F-score using the formula above for beta = 0.5
beta = 0.5
fscore = float((1+beta**2)*((precision*recall)/((beta**2*precision)+recall)))
print(type(fscore))

# Print the results
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))



# Import one metric from sklearn - fbeta_score
from sklearn.metrics import fbeta_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    
    # Calculate the training time
    results['train_time'] = end - start
        
    # Get the predictions on the test set,
    # then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, average='binary', beta=0.5)
        
    # Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, average='binary', beta=0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
    print(results['pred_time'])  
    print(results['train_time'])     
    print(results['acc_train'])
    print(results['acc_test'])
    print(results['f_train'])
    print(results['f_test'])
        
    # Return the results
    return results

# Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

# Initialize the three models
clf_A = GaussianNB()
clf_B = RandomForestClassifier(random_state=0)
clf_C = linear_model.SGDClassifier(loss="log", random_state=0)

# Calculate the number of samples for 1%, 10%, and 100% of the training data

samples_1 = len(X_train.sample(frac=1/100))
print(samples_1)
samples_10 = len(X_train.sample(frac=10/100))
print(samples_10)
samples_100 = len(X_train.index)
print(samples_100)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)


# PARKING LOT
# income = income_raw.apply(lambda x: 1 if x == ">50K" else 0)

# Import the three supervised learning models from sklearn
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.ensemble import AdaBoostClassifier

# TODO: Initialize the three models
# clf_A = GaussianNB()
# clf_B = SVC(random_state=0)
# clf_C = AdaBoostClassifier(random_state=0)


