from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np
import pandas as pd
from numpy import genfromtxt, savetxt
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV


# selective feature array [12, 80, 51, 2, 17, 18, 27, 40]
# range data[:,1:81]

#feature_list = [1, 2, 4, 5, 12, 17, 18, 19, 
#20, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
#40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 56, 59, 61, 62, 66, 67, 68, 69, 70, 71, 75, 76, 77, 
#80]

#{'max_features': 16, 'n_estimators': 2000, 'max_depth': 6, 'min_samples_leaf': 1}

feature_list = [79, 16, 11, 26, 45, 60, 18, 37, 43, 33, 42, 52, 29, 61, 3]

# Random Forest Regressor
def getRFRegressor():
    #return RandomForestRegressor(n_estimators=500, n_jobs=-1)
    return RandomForestRegressor(n_estimators=3000, n_jobs=-1, max_features = 8, random_state = 1, oob_score = True, min_samples_leaf = 1)

# Extreme Random Forest Regressor
def getERFRegressor():
    return ExtraTreesRegressor(n_estimators=2500, n_jobs=-1, max_features = 8, random_state = 1)
    #return ExtraTreesRegressor(n_estimators=1000, n_jobs=-1, max_features = 'sqrt', random_state = 1, oob_score = True, bootstrap = True, min_samples_leaf = 1, min_samples_split = 2)
    #return ExtraTreesRegressor(n_estimators=700, n_jobs=-1, max_features = 17, random_state = 1, oob_score = True, bootstrap = True, max_depth = None, min_samples_split = 2)


def getTrainXData(fileName):
    data = genfromtxt(open(fileName,'r'), delimiter=',', dtype='f8')[1:]
    return data[:,1:81]

def getTrainYData(fileName):
    data = genfromtxt(open(fileName,'r'), delimiter=',', dtype='f8')[1:]
    return data

def getTestXData(fileName):
    data = genfromtxt(open(fileName,'r'), delimiter=',', dtype='f8')[1:]
    return data[:,1:81]

# Collect ID column
def getSubmissionId(fileName):
    return genfromtxt(open(fileName,'r'), delimiter=',', dtype='f8')[1:,:1]

# perform random forest regression
def performRFRegression(train_x_data, train_y_data,test_x_data):
    rf = getERFRegressor()
    rf.fit(train_x_data, train_y_data)
    return rf.predict(test_x_data)[:,None]

# predicted results in kaggle submission format
def saveSubmission(submission_id, test_y, fileName):
    submission_id_and_y = np.append(submission_id,test_y,1)
    savetxt(fileName, submission_id_and_y, header="Id,SalePrice", delimiter=',', comments='', fmt='%i,%f')  

# Mean Square Error and Variance
def calculateMSEAndVariance(train_x_data,train_y_data):
    partial_train_x_data = train_x_data[:-50]
    partial_train_y_data = train_y_data[:-50]

    validation_x_data = train_x_data[-50:]
    validation_y_data = train_y_data[-50:]

    rf = getERFRegressor()
    rf.fit(partial_train_x_data, partial_train_y_data)
    calculated_y = rf.predict(validation_x_data)[:,None]

    return np.mean((calculated_y - validation_y_data) ** 2), rf.score(validation_x_data, validation_y_data)

# Important Features Plot 
def importantFeatures(train_x_data, train_y_data,test_x_data):
    rf = getERFRegressor()
    rf.fit(train_x_data, train_y_data)
    feature_importances =  rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices = np.argsort(feature_importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(train_x_data.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], feature_importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(train_x_data.shape[1]), feature_importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(train_x_data.shape[1]), indices)
    plt.xlim([-1, train_x_data.shape[1]])
    plt.show()

# Hyper parameter tunning (kpitke's implementation)
def hyperparameter_tuning():
    train_x_data = getTrainXData('train_scored.csv')

    train_y_data = getTrainYData('train_scored_y.csv')

    test_x_data = getTestXData('test_scored.csv')

    params = {
              'max_features': [8, 16, 32, 'sqrt', 'auto'],
              'n_estimators': [2500, 3000, 3500, 4000, 4500, 5000]
              }
    est = ExtraTreesRegressor()
    print est.get_params().keys()
    gs_cv = GridSearchCV(est, params)
    gs_cv.fit(train_x_data,train_y_data)
    print gs_cv.best_params_;    


def main():

    train_x_data = getTrainXData('train_scored.csv')

    train_y_data = getTrainYData('train_scored_y.csv')

    test_x_data = getTestXData('test_scored.csv')

    submission_id = getSubmissionId('test_scored.csv')

    test_y = performRFRegression(train_x_data, train_y_data,test_x_data)

    saveSubmission(submission_id, test_y, 'kaggle_submission_with_new_scored_data_and_tuned_algorithm.csv')

    mean_square_error, variance_score  = calculateMSEAndVariance(train_x_data,train_y_data)

    print 'Mean Square Error: ', mean_square_error

    print 'Variance Score: ', variance_score

    hyperparameter_tuning()

    #importantFeatures(train_x_data, train_y_data,test_x_data)


if __name__=="__main__":
    main()