import numpy as np
import knnpredict
import knnrun
import csv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


# TRAINING SET ALLOCATION
# Read training data from csv
fp = open("train50.csv", "r")
reader = csv.reader(fp, delimiter=",")

# Convert training data to numpy array
training50_data = np.array(list(reader))

# Read training data from csv
fp = open("train100.csv", "r")
reader = csv.reader(fp, delimiter=",")

# Convert training data to numpy array
training100_data = np.array(list(reader))

# Read training data from csv
fp = open("train250.csv", "r")
reader = csv.reader(fp, delimiter=",")

# Convert training data to numpy array
training250_data = np.array(list(reader))

# Read training data from csv
fp = open("train500.csv", "r")
reader = csv.reader(fp, delimiter=",")

# Convert training data to numpy array
training500_data = np.array(list(reader))

# Read training data from csv
fp = open("train.csv", "r")
reader = csv.reader(fp, delimiter=",")

# Convert training data to numpy array
training_data = np.array(list(reader))
n, d = training_data.shape

# VALIDATION SET ALLOCATION
# Read validation data from csv
fp = open("validation.csv", "r")
reader = csv.reader(fp, delimiter=",")

# Convert testing data to numpy array
validation_data = np.array(list(reader))
n_validation, d_validation = validation_data.shape

# TESTING SET ALLOCATION
# Read testing data from csv
fp = open("test.csv", "r")
reader = csv.reader(fp, delimiter=",")

# Convert testing data to numpy array
testing_data = np.array(list(reader))
n_test, d_test = testing_data.shape

# Replacement of NA values
training_data = np.where(training_data == '', 1, training_data).astype(float)
training50_data = np.where(training50_data == '', 1, training50_data).astype(float)
training100_data = np.where(training100_data == '', 1, training100_data).astype(float)
training250_data = np.where(training250_data == '', 1, training250_data).astype(float)
training500_data = np.where(training500_data == '', 1, training500_data).astype(float)
validation_data = np.where(validation_data == '', 1, validation_data).astype(float)
testing_data = np.where(testing_data == '', 1, testing_data).astype(float)

# Remove loan default result column, allocate to new array
training_result = training_data[:,d - 1]
training_data = training_data[:,:d - 1]

training50_result = training50_data[:,d - 1]
training50_data = training50_data[:,:d - 1]

training100_result = training100_data[:,d - 1]
training100_data = training100_data[:,:d - 1]

training250_result = training250_data[:,d - 1]
training250_data = training250_data[:,:d - 1]

training500_result = training500_data[:,d - 1]
training500_data = training500_data[:,:d - 1]

validation_result = validation_data[:,d - 1]
validation_data = validation_data[:,:d - 1]

testing_result = testing_data[:,d_test - 1]
testing_data = testing_data[:,:d_test - 1]

#possible k-values
data = [1, 2, 3, 4, 5]

accuracy_sample = []
val = knnrun.run(data, training50_data, validation_data, training50_result, validation_result)
print(1)
accuracy_sample.append(np.mean(validation_result != knnpredict.run(training_data, testing_data, training_result, val)))
val = knnrun.run(data, training100_data, validation_data, training100_result, validation_result)
print(2)
accuracy_sample.append(np.mean(validation_result != knnpredict.run(training_data, testing_data, training_result, val)))
val = knnrun.run(data, training250_data, validation_data, training250_result, validation_result)
print(3)
accuracy_sample.append(np.mean(validation_result != knnpredict.run(training_data, testing_data, training_result, val)))
val = knnrun.run(data, training500_data, validation_data, training500_result, validation_result)
print(4)
accuracy_sample.append(np.mean(validation_result != knnpredict.run(training_data, testing_data, training_result, val)))
val = knnrun.run(data, training_data, validation_data, training_result, validation_result)
print(5)
temp = np.mean(validation_result != knnpredict.run(training_data, testing_data, training_result, val))
accuracy_sample.append(temp)
print(accuracy_sample)
second_sample = []
for k in range(len(data)):
    modeo = knnpredict.run(training_data, testing_data, training_result, data[k])
    ert = np.mean(testing_result != modeo)
    second_sample.append(ert)
plt.figure()
plt.plot([50, 100, 250, 500, 1000], accuracy_sample)
plt.title('KNN Accuracy vs. Training Set Size')
plt.ylabel('Error Rate')
plt.xlabel('Training Set Size')
plt.savefig("KNN_sample.png")
plt.figure()
plt.plot([1, 2, 3, 4, 5], second_sample)
plt.title('Accuracy vs. K-Value')
plt.ylabel('Error Rate')
plt.xlabel('K-Value')
plt.savefig("KNN_value.png")
print(temp)
