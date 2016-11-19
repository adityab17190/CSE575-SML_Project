from sklearn.ensemble import RandomForestRegressor
import numpy as np
from numpy import genfromtxt, savetxt

# selective feature array [12, 80, 51, 2, 17, 18, 27, 40]

def getTrainXData(fileName):
    data = genfromtxt(open(fileName,'r'), delimiter=',', dtype='f8')[1:]
    return data[:,1:81]

def getTrainYData(fileName):
    data = genfromtxt(open(fileName,'r'), delimiter=',', dtype='f8')[1:]
    return data[:,81:]

def getTestXData(fileName):
    data = genfromtxt(open(fileName,'r'), delimiter=',', dtype='f8')[1:]
    return data[:,1:81]

# Collect ID column
def getSubmissionId(fileName):
    return genfromtxt(open(fileName,'r'), delimiter=',', dtype='f8')[1:,:1]

# perform random forest regression
def performRFRegression(train_x_data, train_y_data,test_x_data):
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(train_x_data, train_y_data)
    return rf.predict(test_x_data)[:,None]

# predicted results in kaggle submission format
def saveSubmission(submission_id, test_y, fileName):
    submission_id_and_y = np.append(submission_id,test_y,1)
    savetxt(fileName, submission_id_and_y, header="Id,SalePrice", delimiter=',', comments='', fmt='%i,%f')

# Mean Square Error and Variance
def calculateMSEAndVariance(train_x_data,train_y_data):
    partial_train_x_data = train_x_data[:-100]
    partial_train_y_data = train_y_data[:-100]

    validation_x_data = train_x_data[-100:]
    validation_y_data = train_y_data[-100:]

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(partial_train_x_data, partial_train_y_data)
    calculated_y = rf.predict(validation_x_data)[:,None]

    return np.mean((calculated_y - validation_y_data) ** 2), rf.score(validation_x_data, validation_y_data)


def main():

    train_x_data = getTrainXData('train_scored.csv')

    train_y_data = getTrainYData('train_scored.csv')

    test_x_data = getTestXData('test_scored.csv')

    submission_id = getSubmissionId('test_scored.csv')

    test_y = performRFRegression(train_x_data, train_y_data,test_x_data)

    saveSubmission(submission_id, test_y, 'kaggle_submission.csv')

    mean_square_error, variance_score  = calculateMSEAndVariance(train_x_data,train_y_data)

    print 'Mean Square Error: ', mean_square_error

    print 'Variance Score: ', variance_score


if __name__=="__main__":
    main()