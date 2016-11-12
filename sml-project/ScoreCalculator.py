__author__ = 'aditya'

import csv

import  numpy

from sklearn import  linear_model

trainDataFile='./data/train.csv';

discreetFeaturesFile='./data/discreet.txt';

mapping=dict()

discreetFeatures=[]

totalArea=[]

lotSizeValues=[]

perSqFeet=[]

salePrice=[]

scoredNeighborhood=[]

cond1Scored=[]

cond2Scored=[]

mszoningScored=[]

conditionScored=[]

garageTypeScored=[]

exterior1stValues=[]

exterior1stScored=[]

firePlaceValues=[]

fullbathValues=[]

basementArea=[]

firstFlrArea=[]

secondFlrArea=[]

garageArea=[]

livingArea=[]

garageCars=[]

garageTypeValues=[]

misc=[]

ageValues=[]

overallCondition=[]

overallQuality=[]

neighborhoodValues=[]

condition1Values=[]

condition2Values=[]

mszoningValues=[]

bedroomAboveGradeValues=[]

totalRoomsAboveGradeValues=[]

yearSoldValues=[]

exterQualValues=[]

exterCondValues=[]

exterQualScored=[]

exterCondScored=[]

garageQualValues=[]

garageCondValues=[]

garageQualScored=[]

garageCondScored=[]

kitchenQualityValues=[]

kitchenQualityScored=[]

heatingQualityValues=[]

heatingQualityScored=[]


scoredMappingDict=dict()



def readTrainData():

  file=open(trainDataFile)

  reader=csv.reader(file)

  header=next(reader)

  n=len(header)

  for i in range(0,n):
      mapping[header[i]]=i
      print(header[i])

  #print(mapping)



def readDiscreetFeatures():

    file=open(discreetFeaturesFile)

    for feature in file:
        feature=feature[:-1]
        discreetFeatures.append(feature)

    print(discreetFeatures)



def getAreaAndPrice():

  file=open(trainDataFile)

  reader=csv.reader(file)

  header=next(reader)

  n=len(header)

  lotSizeIndex=mapping['LotArea']

  basementAreaIndex=mapping['TotalBsmtSF']

  firstFloorAreaIndex=mapping['1stFlrSF']

  secondFloorAreaIndex=mapping['2ndFlrSF']

  livingAreaIndex=mapping['GrLivArea']

  garageAreaIndex=mapping['GarageArea']

  salePriceIndex=mapping['SalePrice']

  miscIndex=mapping['MiscVal']

  yearBuiltIndex=mapping['YearBuilt']

  yearRemodIndex=mapping['YearRemodAdd']

  overallQualIndex=mapping['OverallQual']

  overallCondIndex=mapping['OverallCond']

  neighborhoodIndex=mapping['Neighborhood']

  condition1Index=mapping['Condition1']

  condition2Index=mapping['Condition2']

  mszoningIndex=mapping['MSZoning']

  bedroomAboveGradeIndex=mapping['BedroomAbvGr']

  totalRmsAboveGradeIndex=mapping['TotRmsAbvGrd']

  yearSoldIndex=mapping['YrSold']

  garageCarIndex=mapping['GarageCars']

  garageTypeIndex=mapping['GarageType']

  firePlaceIndex=mapping['Fireplaces']

  fullbathIndex=mapping['FullBath']

  exterior1stIndex=mapping['Exterior1st']

  exterQualIndex=mapping['ExterQual']

  exterCondIndex=mapping['ExterCond']

  garageQualIndex=mapping['GarageQual']

  garageCondIndex=mapping['GarageCond']

  kitchenQualIndex=mapping['KitchenQual']

  heatingQualIndex=mapping['HeatingQC']

  for row in reader:

      basementArea.append(float(row[basementAreaIndex]))

      firstFlrArea.append(float(row[firstFloorAreaIndex]))

      secondFlrArea.append(float(row[secondFloorAreaIndex]))

      livingArea.append(float(row[livingAreaIndex]))

      garageArea.append(float(row[garageAreaIndex]))

      total=float(row[basementAreaIndex])+float(row[firstFloorAreaIndex])+float(row[secondFloorAreaIndex])+float(row[livingAreaIndex])+float(row[garageAreaIndex]);

      totalArea.append(total)

      lotArea=float(row[lotSizeIndex])

      lotSizeValues.append(lotArea)

      misc.append(float(row[miscIndex]))

      price=float(row[salePriceIndex])

      salePrice.append(price)

      #perSqFeet.append(price/lotArea)
      perSqFeet.append(price/total)

      yearBuilt=int(row[yearBuiltIndex])

      yearRemod=int(row[yearRemodIndex])

      a=2016-max(yearBuilt,yearRemod)

      ageValues.append(a)

      overallQuality.append(int(row[overallQualIndex]))

      overallCondition.append(int(row[overallCondIndex]))

      neighborhoodValues.append(row[neighborhoodIndex])

      condition1Values.append(row[condition1Index])

      condition2Values.append(row[condition2Index])

      bedroomAboveGradeValues.append(int(row[bedroomAboveGradeIndex]))

      totalRoomsAboveGradeValues.append(int(row[totalRmsAboveGradeIndex]))

      yearSoldValues.append(int(row[yearSoldIndex]))

      mszoningValues.append(row[mszoningIndex])

      garageCars.append(int(row[garageCarIndex]))

      garageTypeValues.append(row[garageTypeIndex])

      firePlaceValues.append(int(row[firePlaceIndex]))

      fullbathValues.append(int(row[fullbathIndex]))

      exterior1stValues.append(row[exterior1stIndex])

      exterQualValues.append(row[exterQualIndex])

      exterCondValues.append(row[exterCondIndex])

      garageQualValues.append(row[garageQualIndex])

      garageCondValues.append(row[garageCondIndex])

      kitchenQualityValues.append(row[kitchenQualIndex])

      heatingQualityValues.append(row[heatingQualIndex])


  scoredMappingDict['Neighborhood']=neighborhoodValues

  scoredMappingDict['Condition1']=condition1Values

  scoredMappingDict['Condition2']=condition2Values

  scoredMappingDict['MSZoning']=mszoningValues

  scoredMappingDict['GarageType']=garageTypeValues

  scoredMappingDict['Exterior1st']=exterior1stValues

  scoredMappingDict['ExterQual']=exterQualValues

  scoredMappingDict['ExterCond']=exterCondValues

  scoredMappingDict['GarageQual']=garageQualValues

  scoredMappingDict['GarageCond']=garageCondValues

  scoredMappingDict['KitchenQual']=kitchenQualityValues

  scoredMappingDict['HeatingQC']=heatingQualityValues

#  print(totalArea)

  #print(lotSizeValues)

  #print(salePrice)

  #print(perSqFeet)


#def computePerSqFoot():


def assignScore(column):

  values=scoredMappingDict[column]

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

      #print("Key:"+str(key)+" Value="+str(avgPrice))

  #print(columnToAvgPriceMapping)

  #Normalizing

  #denominator=0.0

  #for key in columnToAvgPriceMapping:

   #   denominator=denominator+columnToAvgPriceMapping[key]

  columnValueToScoreMapping=dict()


  for key in columnToAvgPriceMapping:

      columnValueToScoreMapping[key]=columnToAvgPriceMapping[key]/minimum

    #  print(key+":"+str(columnValueToScoreMapping[key]))



  scoredValues=[]

  for value in values:

      scoredValues.append(columnValueToScoreMapping[value])

  #print("Scored "+column)

  #print(scoredValues)

  return scoredValues


def linearRegression():

    x=[]

    n=len(salePrice)
    for i in range(0,n):

       # x.append((scoredNeighborhood[i],lotSizeValues[i],basementArea[i],firstFlrArea[i],secondFlrArea[i],livingArea[i],garageArea[i]))

       x.append((scoredNeighborhood[i],totalArea[i],overallQuality[i],overallCondition[i],bedroomAboveGradeValues[i],exterQualScored[i],exterCondScored[i]))

    y=salePrice

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


def readTestData():

  file=open(trainDataFile)

  reader=csv.reader(file)

  header=next(reader)

  n=len(header)

  neighborhoodIndex=mapping['Neighborhood']

  basementAreaIndex=mapping['TotalBsmtSF']

  firstFloorAreaIndex=mapping['1stFlrSF']

  secondFloorAreaIndex=mapping['2ndFlrSF']

  livingAreaIndex=mapping['GrLivArea']

  garageAreaIndex=mapping['GarageArea']

  test_data=[]

  for row in reader:

      neighborhood=row[neighborhoodIndex]

      score=scoredNeighborhood[neighborhood]

      total=float(row[basementAreaIndex])+float(row[firstFloorAreaIndex])+float(row[secondFloorAreaIndex])+float(row[livingAreaIndex])+float(row[garageAreaIndex]);

      test_data.append((score,total))


  return test_data


def computeAvgConditionScored():

    conditionScored=[]
    n=len(cond1Scored)

    for i in range(0,n):
        conditionScored.append((cond1Scored[i]+cond2Scored[i])/2)

    return conditionScored


if __name__ == '__main__':
    readTrainData()
    readDiscreetFeatures()
    getAreaAndPrice()
    scoredNeighborhood=assignScore('Neighborhood')
    cond1Scored=assignScore('Condition1')
    cond2Scored=assignScore('Condition2')
    mszoningScored=assignScore('MSZoning')
    garageTypeScored=assignScore('GarageType')
    exterior1stScored=assignScore('Exterior1st')
    exterQualScored=assignScore('ExterQual')
    exterCondScored=assignScore('ExterCond')
    garageQualScored=assignScore('GarageQual')
    garageCondScored=assignScore('GarageCond')
    kitchenQualityScored=assignScore('KitchenQual')
    heatingQualityScored=assignScore('HeatingQC')
    conditionScored=computeAvgConditionScored()
    scaledArea=[x/100 for x in totalArea]
    linearRegression()
#    linearRegression(scoredNeighborhood,totalArea)

