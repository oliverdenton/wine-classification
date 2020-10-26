# Supervised Learning Wine Classification
Using supervised learning and Bayesian classifier to determine the type of grape used in a given wine. A leave-one-out approach was taken to test and train the model. This is where:
  * The first sample from the data set is used for testing
  * The model is trained using all the remaining N - 1 samples from the data set
  * This is repeated using the second sameple and so forth until all N samples have been tested

In the end, we will have trained N different classifiers. This allows us use sufficient data to test and train the classifier and estimate its parameters more accurately. As opposed to splitting the data set in half, as seen in code omitted from classifier.py.

## The Data Set
The data set used can be found at https://archive.ics.uci.edu/ml/datasets/wine and contains 178 samples each with 13 measurements.
