__author__ = 'aditya'
import csv

import sys

mapping=dict() #Maps Features to their respective column numbers

continuousFeatures=[]

continuousFeaturesIndices=[]

trainDataFile='./data/train.csv'

testDataFile='./data/test.csv'

continuousFeaturesFile='./data/Continuous.txt'

featureToValuesMappingTrain=dict()

featureToValuesMappingTest=dict()

globalFeatureToScoresMapping=dict()

salePriceTrain=[]

header=[]

trainDataLen=0

testDataLen=0


def assignScores(values,perSqFeet):

      #print(perSqFeet)

      columnValueToSalePriceListMapping=dict()

      n=len(values)


      for i in range(0,n):

          columnValue=values[i]

          price=perSqFeet[i]

          priceList=columnValueToSalePriceListMapping.get(columnValue,None)

          if priceList is None:
              priceList=[]

          priceList.append(price)

          columnValueToSalePriceListMapping[columnValue]=priceList

      columnToAvgPriceMapping=dict()

      minimum=float("inf")

      for key in columnValueToSalePriceListMapping:

        priceList=columnValueToSalePriceListMapping[key]

        priceSum=sum(priceList)

        count=len(priceList)

        avgPrice=priceSum/count

        if avgPrice<minimum:
            minimum=avgPrice

        columnToAvgPriceMapping[key]=avgPrice

    #  print ('Key:'+str(key)+' Avg:'+str(avgPrice))


      columnValueToScoreMapping=dict()


      for key in columnToAvgPriceMapping:

        columnValueToScoreMapping[key]=columnToAvgPriceMapping[key]/minimum

     #   print('Key:'+str(key)+' Value:'+str(columnValueToScoreMapping[key]))


      return columnValueToScoreMapping



def generateScoredData(columnValueToScoreMapping,values):

      scoredValues=[]

      for value in values:

        scoredValues.append(columnValueToScoreMapping.get(value,0))

      return scoredValues



def mapColumnNamesToColumnNumbers():

      print('.........................................................................')

      print('Mapping features to column numbers...')

      file=open(trainDataFile)

      reader=csv.reader(file)

      header=next(reader)

      n=len(header)

      for i in range(0,n):
        mapping[header[i]]=i
        if header[i] in continuousFeatures:
            continuousFeaturesIndices.append(i)

        print(header[i]+":"+str(i))

      file.close()

      return header


def readContinuousFeatures():

    file=open(continuousFeaturesFile)

    for feature in file:
        if feature.endswith('\n'):
            feature=feature[:-1]
        continuousFeatures.append(feature)


    file.close()


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

            values.append(row[i])

            featureToValuesMapping[feature]=values

    return count



def computeContinuousFeatures(isTrainingData):

    if isTrainingData:
        featureToValuesMapping=featureToValuesMappingTrain
    else:
        featureToValuesMapping=featureToValuesMappingTest

    for key in featureToValuesMapping:

        if key not in continuousFeatures:
            continue
        print(key)
        values=featureToValuesMapping[key]
        newValues=[]

        for value in values:
            if value=='NA':
                newValues.append(0)
            else:
                newValues.append(float(value))
        featureToValuesMapping[key]=newValues

def computePerSqFootPrice():

    totalArea=featureToValuesMappingTrain['TotalArea']

    salePrice=featureToValuesMappingTrain['SalePrice']

    perSqFoot=[]

    n=len(totalArea)

    for i in range(0,n):
        perSqFoot.append(salePrice[i]/totalArea[i])

    featureToValuesMappingTrain['PerSqFootPrice']=perSqFoot


def computeScores():
    print('Computing Scores')

    perSqFootPrice=featureToValuesMappingTrain['PerSqFootPrice']

    for key in featureToValuesMappingTrain:

        if key in continuousFeatures:
            continue
        else:
            print(key)
            values=featureToValuesMappingTrain[key]
            scoresMap=assignScores(values,perSqFootPrice)
            globalFeatureToScoresMapping[key]=scoresMap


def insertScoredData(isTrainingData):

     print('Inserting Scored Data')

     if isTrainingData:
        featureToValuesMapping=featureToValuesMappingTrain
     else:
        featureToValuesMapping=featureToValuesMappingTest

     for key in featureToValuesMapping:
         if key in continuousFeatures:
             continue
         else:
             values=featureToValuesMapping[key]
             print(key)
             newValues=generateScoredData(globalFeatureToScoresMapping[key],values)
             featureToValuesMapping[key]=newValues

def writeScoredDataToFile(isTrainData):

    if isTrainData:
        filePath='./data/train_scored.csv'
        id=0
        n=trainDataLen
        featureToValuesMapping=featureToValuesMappingTrain
    else:
        filePath='./data/test_scored.csv'
        id=1460
        n=testDataLen
        featureToValuesMapping=featureToValuesMappingTest

    if sys.version_info[0] > 3:
        print ('Above 3')
        file=open(filePath,'w',newline='')
    else:
        file=open(filePath,'w')

    writer=csv.writer(file)

    writer.writerow(header)

    headers=len(header)

    if not isTrainData:
        headers=headers-1

    print('n='+str(n))

    print('Headers='+str(headers))

    for i in range(0,n):
        id=id+1
        line=[]
        line.append(id)
        for j in range(1,headers):
            print(header[j])
            values=featureToValuesMapping[header[j]]
            line.append(values[i])
        writer.writerow(line)

    file.close()

if __name__ == '__main__':

    readContinuousFeatures()
    header=mapColumnNamesToColumnNumbers()
    trainDataLen=readData(True)
    testDataLen=readData(False)
    computeContinuousFeatures(True)
    computeContinuousFeatures(False)
    computePerSqFootPrice()
    computeScores()
    insertScoredData(True)
    insertScoredData(False)
    writeScoredDataToFile(True)
    writeScoredDataToFile(False)












