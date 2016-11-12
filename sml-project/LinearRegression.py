__author__ = 'aditya'
import csv

import numpy

import math

from sklearn import  linear_model

import ScoringScript

trainDataFile='./data/train.csv'

testDateFile='./data/test.csv'

featuresToIncludeFile='./data/include.csv'

featuresToScoreFile='./data/score.txt'



mapping=dict() #Maps Features to their respective column numbers

featuresToInclude=[]

columnsToInclude=[]

featureToTypeMapping=dict()

featuresToScore=[]

featureToValuesMappingTraining=dict()

featureToValuesMappingTest=dict()

totalAreaTrain=[]

totalAreaTest=[]

salePriceTrain=[]

perSqFootTrain=[]

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

def readFeaturesToInclude():


    print('.........................................................................')

    print('Reading Features to include and their types')

    file=open(featuresToIncludeFile)

    reader=csv.reader(file)

    next(reader) #Skipping first line

    for line in reader:
        featuresToInclude.append(line[0])
        featureToTypeMapping[line[0]]=line[1]
        columnsToInclude.append(mapping[line[0]])
        print(line[0]+":"+line[1]+':'+str(mapping[line[0]]))

    file.close()

def readFeaturesToScore():


    print('.........................................................................')

    print('Reading Features to Score..')

    file=open(featuresToScoreFile)

    for feature in file:
        if feature.endswith('\n'):
            feature=feature[:-1]
        featuresToScore.append(feature)
        print(feature)

    file.close()

def readData(isTrainData):


    print('.........................................................................')

    if isTrainData:

        print('Reading Train Data')
    else:
        print('Reading Test Data')

    path=trainDataFile

    if not isTrainData:
        path=testDateFile

    print(path)


    file=open(path)

    reader=csv.reader(file)

    header=next(reader)

    print(header)

    print(len(header))

    featureToValuesMapping=featureToValuesMappingTraining

    if not isTrainData:
        featureToValuesMapping=featureToValuesMappingTest

    for row in reader:
        for i in columnsToInclude:
           # print(i)
            feature=header[i]
            values=featureToValuesMapping.get(feature,None)
            if values is None:
                values=[]
            values.append(row[i])
            featureToValuesMapping[feature]=values

        if isTrainData:
            salePriceTrain.append(float(row[len(row)-2]))

    file.close()

def preProcessData(featureToValuesMapping):

    print('.........................................................................')

    print('Preprocessing data')

    for feature in featureToValuesMapping:

        datatype=featureToTypeMapping[feature]

        print(feature)

        if datatype =='String':
            continue
        else:
            values=featureToValuesMapping[feature]
            processedValues=[float(x) for x in values ]
            featureToValuesMapping[feature]=processedValues

        print(featureToValuesMapping[feature])

def computePerSqFootPrice():


    print('.........................................................................')

    print('Computing per square foot price..')

    totalArea=featureToValuesMappingTraining['TotalArea']

    n=len(totalArea)

    for i in range(0,n):
        perSqFootTrain.append(salePriceTrain[i]/totalArea[i])

    print(perSqFootTrain)

def getScoredValues(isTrainingData):

    if isTrainingData:

        featureToValuesMapping=featureToValuesMappingTraining
    else:
        featureToValuesMapping=featureToValuesMappingTest

    for feature in featuresToScore:
    #    print(feature)
        scoredValues=ScoringScript.computeScoredData(featureToValuesMapping[feature],perSqFootTrain)
        featureToValuesMapping[feature]=scoredValues
    #    print(scoredValues)


def performLinearRegression():

    x=[]

    n=len(salePriceTrain)

    for i in range(0,n):
        l=[]
        for feature in featuresToInclude:
            values=featureToValuesMappingTraining[feature]
            l.append(values[i])
        x.append(tuple(l))

    #print(x)

    y=salePriceTrain

    regr=linear_model.LinearRegression()

    x_train=x[:-140]

    x_test=x[-140:]

    y_train=y[:-140]

    y_test=y[-140:]

    regr.fit(x_train,y_train)

    print('Coefficients')

    print(regr.coef_)

    print("Mean squared error: %.2f"
      % numpy.mean((regr.predict(x_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x_test, y_test))



if __name__ == '__main__':
    mapColumnNamesToColumnNumbers()
    readFeaturesToInclude()
    readFeaturesToScore()
    readData(True)
    readData(False)
    preProcessData(featureToValuesMappingTraining)
#    preProcessData(featureToValuesMappingTest)
    if 'TotalArea' in featuresToInclude:
        computePerSqFootPrice()
        getScoredValues(True)
    #getScoredValues(False)

    performLinearRegression()
    exit(0)


