from sklearn.svm import SVC
import numpy as np
from numpy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pcalearn
import pcaproj
import csv

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
training_data = np.where(training_data == '', 1, training_data).astype(np.float)
training50_data = np.where(training50_data == '', 1, training50_data).astype(np.float)
training100_data = np.where(training100_data == '', 1, training100_data).astype(np.float)
training250_data = np.where(training250_data == '', 1, training250_data).astype(np.float)
training500_data = np.where(training500_data == '', 1, training500_data).astype(np.float)
validation_data = np.where(validation_data == '', 1, validation_data).astype(np.float)
testing_data = np.where(testing_data == '', 1, testing_data).astype(np.float)

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






# Initialize PCA training model and creating projection matrix
F = 30
mu, Z = pcalearn.run(F, validation_data)
mu = np.empty(d)

# Tuning F Dimensionality through explained variance ratio
cov_matrix = np.cov(Z, rowvar=False)
egnvalues, egnvectors = eigh(cov_matrix)
total_egnvalues = sum(egnvalues)
var_exp = [(i / total_egnvalues) for i in sorted(egnvalues, reverse=True)]

# Scree Plot formation and export to scree.png
plt.figure()
plt.plot(range(0,len(var_exp)), var_exp)
plt.title('PCA Scree Plot')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.savefig("scree.png")





# Scree Optimal F-value
F = 6

# Plot Accuracy vs. F-value
accuracy_hyper = []
for i in range(1, 30):
    mu, Z = pcalearn.run(i, training_data)
    training_data_small = pcaproj.run(training_data, mu, Z)
    testing_data_small = pcaproj.run(testing_data, mu, Z)

    # Fitting model against testing set
    alg = SVC(C=10, kernel='rbf', gamma=1000.0)
    alg.fit(training_data_small, training_result)
    testing_pred = alg.predict(testing_data_small)

    accuracy_hyper.append(np.mean(testing_result != testing_pred))

plt.figure()
plt.plot(range(0,len(accuracy_hyper)), accuracy_hyper)
plt.title('PCA Accuracy vs. Principle Component Index')
plt.ylabel('Percent Error')
plt.xlabel('Principal component index')
plt.savefig("accuracy_hyper.png")
    



# Plot Accuracy vs. sample-size
accuracy_sample = []

mu, Z = pcalearn.run(F, training50_data)
training_data_small = pcaproj.run(training50_data, mu, Z)
testing_data_small = pcaproj.run(testing_data, mu, Z)

# Fitting model against testing set (size 50)
alg = SVC(C=10, kernel='rbf', gamma=1000.0)
alg.fit(training_data_small, training50_result)
testing_pred = alg.predict(testing_data_small)

accuracy_sample.append(np.mean(testing_result != testing_pred))

mu, Z = pcalearn.run(F, training100_data)
training_data_small = pcaproj.run(training100_data, mu, Z)
testing_data_small = pcaproj.run(testing_data, mu, Z)

# Fitting model against testing set (size 100)
alg = SVC(C=10, kernel='rbf', gamma=1000.0)
alg.fit(training_data_small, training100_result)
testing_pred = alg.predict(testing_data_small)

accuracy_sample.append(np.mean(testing_result != testing_pred))

mu, Z = pcalearn.run(F, training250_data)
training_data_small = pcaproj.run(training250_data, mu, Z)
testing_data_small = pcaproj.run(testing_data, mu, Z)

# Fitting model against testing set (size 250)
alg = SVC(C=10, kernel='rbf', gamma=1000.0)
alg.fit(training_data_small, training250_result)
testing_pred = alg.predict(testing_data_small)

accuracy_sample.append(np.mean(testing_result != testing_pred))

mu, Z = pcalearn.run(F, training500_data)
training_data_small = pcaproj.run(training500_data, mu, Z)
testing_data_small = pcaproj.run(testing_data, mu, Z)

# Fitting model against testing set (size 500)
alg = SVC(C=10, kernel='rbf', gamma=1000.0)
alg.fit(training_data_small, training500_result)
testing_pred = alg.predict(testing_data_small)

accuracy_sample.append(np.mean(testing_result != testing_pred))

# Generating PCA model from training set
mu, Z = pcalearn.run(F, training_data)
training_data_small = pcaproj.run(training_data, mu, Z)
testing_data_small = pcaproj.run(testing_data, mu, Z)

# Fitting model against testing set (size 1000)
alg = SVC(C=10, kernel='rbf', gamma=1000.0)
alg.fit(training_data_small, training_result)
testing_pred = alg.predict(testing_data_small)

accuracy_sample.append(np.mean(testing_result != testing_pred))

plt.figure()
plt.plot([50, 100, 250, 500, 1000], accuracy_sample)
plt.title('PCA Accuracy vs. Training Set Size')
plt.ylabel('Percent Error')
plt.xlabel('Training Set Size')
plt.savefig("accuracy_sample.png")





# Generating PCA model from training set
mu, Z = pcalearn.run(F, training_data)
training_data_small = pcaproj.run(training_data, mu, Z)
testing_data_small = pcaproj.run(testing_data, mu, Z)

# Fitting model against testing set
alg = SVC(C=10, kernel='rbf', gamma=1000.0)
alg.fit(training_data_small, training_result)
testing_pred = alg.predict(testing_data_small)

# Computing percent error for F = 6
print(np.mean(testing_result != testing_pred))