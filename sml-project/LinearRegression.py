__author__ = 'aditya'

import csv

import sys

from sklearn import linear_model

import numpy

# trainDataFile='./data/PCA_train_scored.csv'
# testDataFile='./data/PCA_test_scored.csv'
# featuresToIncludeFile='./data/PCA_includedFeatures.txt'


trainDataFile='./data/feature_engineering_test/train_new_scored.csv'
testDataFile='./data/feature_engineering_test/test_new_scored.csv'
featuresToIncludeFile='./data/includedFeatures.txt'

yFile = './data/train_scored_y.csv';

mapping=dict()

featureToValuesMappingTrain=dict()

featureToValuesMappingTest=dict()

featuresToInclude=[]


def mapColumnNamesToColumnNumbers():

      print('.........................................................................')

      print('Mapping features to column numbers...')

      file=open(trainDataFile)

      reader=csv.reader(file)

      header=next(reader)

      n=len(header)

      for i in range(0,n):
        mapping[header[i]]=i

        print(header[i]+":"+str(i))

      file.close()

      return header


def readData(isTrainingData):

    if isTrainingData:
        featureToValuesMapping=featureToValuesMappingTrain
        file=open(trainDataFile)

    else:
        featureToValuesMapping=featureToValuesMappingTest
        file=open(testDataFile)


    reader=csv.reader(file)

    header=next(reader) #Skipping header

    n=len(header)

    if not isTrainingData:
        n=n-1

    count=0

    for row in reader:

        count=count+1

        for i in range(1,n):

            feature=header[i]

            values=featureToValuesMapping.get(feature,None)

            if values is None:
                values=[]

            values.append(float(row[i]))

            featureToValuesMapping[feature]=values

    return count


def readFeaturesToInclude():

    file=open(featuresToIncludeFile)

    for feature in file:

        if feature.endswith('\n'):
            feature=feature[:-1]
        print(feature)
        featuresToInclude.append(feature)


def genrateTrainAndTestData(trainDataLen,testDataLen):

    # y_train=featureToValuesMappingTrain['SalePrice']

    file=open(yFile)
    reader=csv.reader(file)
    next(reader)
    y_train = []

    for row in reader:
        price = float(row[0])
        y_train.append(price)

    x_train=getFeatures(featureToValuesMappingTrain,trainDataLen)

    x_test=getFeatures(featureToValuesMappingTest,testDataLen)

    return x_train,y_train,x_test


def performLinearRegression(trainDataLen,testDataLen):

    x_train,y_train,x_test=genrateTrainAndTestData(trainDataLen,testDataLen)

    trainingData=x_train[:-140]

    trainingY=y_train[:-140]

    validationData=x_train[-140:]

    validationY=y_train[-140:]

    regr=linear_model.LinearRegression()

    regr.fit(trainingData,trainingY)

    print('Coefficients')

    print(regr.coef_)

    coefficients=regr.coef_

    print("Mean squared error: %.2f"
      % numpy.mean((regr.predict(validationData) - validationY) ** 2))
# Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(validationData, validationY))

    predictedSalePrices=predict(coefficients,x_test)

    writeOutput(predictedSalePrices)


def predict(coefficients,testData):

    noOfFeatures=len(coefficients)

    n=len(testData)

    predicted=[]

    for i in range(0,n):
        y=0
        l=list(testData[i])
        for j in range(0,noOfFeatures):
            y=y+coefficients[j]*l[j]
        predicted.append(y)

    return predicted


def writeOutput(predictedValues):

    n=len(predictedValues)

    id=1460

    filePath='./data/output.csv'

    if sys.version_info[0] > 3:

        file=open(filePath,'w',newline='')
    else:
        file=open(filePath,'w')

    writer=csv.writer(file)

    header=['Id','SalePrice']

    writer.writerow(header)

    for i in range(0,n):
        id=id+1
        line=[]
        line.append(id)
        line.append(predictedValues[i])
        writer.writerow(line)

    file.close()


def getFeatures(featureToValuesMapping, dataLen):

    x=[]

    for i in range(0,dataLen):
        l=[]
        for feature in featuresToInclude:
            values=featureToValuesMapping[feature]
            l.append(values[i])

        x.append(l)
    return  x

if __name__ == '__main__':
    mapColumnNamesToColumnNumbers()
    trainDataLen=readData(True)
    testDataLen=readData(False)
    readFeaturesToInclude()
    performLinearRegression(trainDataLen,testDataLen)