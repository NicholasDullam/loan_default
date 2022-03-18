# Loan Default Prediction

### What is this?
A training, testing, and validation approach to predicting the defaulting of loans based upon 50 features provided a pruned sample of individual financial statements for CS 373. We evaluated the data through two means: Principle component analysis and k-nearest neighbors. With all implementations through python, we tuned our hyperparameters for PCA to include a dimensionality reduction of 6, minimizing our error rate of ~8.4%. As for our k-nearest neighbors implementation, we minimized our error rate over sample sizes of 100, with k-values of 2 and 4 respectively, at ~10%
### Getting started
To install all relevant dependencies run...
```
pip install -r requirements.txt
```
To preprocess the sample data run...
```
cd sources
python preprocess.py
```
To run principle component analysis testing run...
```
cd sources
python pcamain.py
```
To run k-nearest-neighbors testing run...
```
cd sources
python knnmain.py
```
