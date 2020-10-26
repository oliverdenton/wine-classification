import numpy as np
import matplotlib.pyplot as plt

#Assumption that data can be modelled as a Guassian distribution
from scipy.stats import multivariate_normal

#Load data and separate by class of grape
X = np.loadtxt(open("data/wines.txt", "r"), delimiter=',')
# wines1 = X[X[:, 0]==1, :]
# wines2 = X[X[:, 0]==2, :]
# wines3 = X[X[:, 0]==3, :]

def test_loop(data, test_index):
    #Trains classifer using all but one datapoint (test_index)
    #Tests the same test_index and return whether or not correct

    #Leave-one-out testing
    train_data = np.delete(data, test_index, axis=0)
    test_data = data[test_index, :]

    #Separate by class of grape
    train1 = train_data[train_data[:, 0] == 1]
    train2 = train_data[train_data[:, 0] == 2]
    train3 = train_data[train_data[:, 0] == 3]

    #Estimate mean and covariance for each class (training stage)
    mean1 = np.mean(train1[:, 1:], axis=0)
    mean2 = np.mean(train2[:, 1:], axis=0)
    mean3 = np.mean(train3[:, 1:], axis=0)
    #Rowvar set to 0 as columns = vars, rows = observations
    cov1 = np.cov(train1[:, 1:], rowvar=0)
    cov2 = np.cov(train2[:, 1:], rowvar=0)
    cov3 = np.cov(train3[:, 1:], rowvar=0)

    #Create distribution objects for each class with parameters
    dist1 = multivariate_normal(mean=mean1, cov=cov1)
    dist2 = multivariate_normal(mean=mean2, cov=cov2)
    dist3 = multivariate_normal(mean=mean3, cov=cov3)

    #Evaluate each probability density funciton for all test data
    p1 = dist1.pdf(test_data[1:])
    p2 = dist2.pdf(test_data[1:])
    p3 = dist3.pdf(test_data[1:])
    probabilities = np.vstack((p1, p2, p3))

    #Compute most likely class of grape, check whether correct
    index = np.argmax(probabilities, axis=0) + 1
    correct = index == data[test_index, 0]
    return correct

def classify(data):
    #Classifies every sample using leave-one-out training
    ncorrect = 0
    ntotal = data.shape[0]

    #Train and test classifier N times
    for index in range(ntotal):
        ncorrect = ncorrect + test_loop(data, index)
    
    #Evaluate 'correctness' of classifer
    percent_correct = ncorrect * 100.0 / ntotal
    return str(percent_correct[0]) + "% correct"

print(classify(X))


#Split into equal testing and training partitions
#using even/odd lines as test/training data
# wines1_test = wines1[0::2, :]
# wines1_train = wines1[1::2, :]
# wines2_test = wines2[0::2,:]
# wines2_train = wines2[1::2, :]
# wines3_test = wines3[0::2, :]
# wines3_train = wines3[1::2, :]
# wines_test= np.vstack((wines1_test, wines2_test, wines3_test))

#Plotting classification makes it possible to visually spot errors
# plt.plot(classes, 'k.-', ms=10)
# plt.show()

#Calculate the classification error as a percentage
# correct = wines_test[:, 0] == classes
# percent_correct = np.sum(correct) * 100.0 / classes.shape
# print(percent_correct)