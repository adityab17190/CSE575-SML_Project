__author__ = 'aditya'

def computeScoredData(values,perSqFeet):

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

        print ('Key:'+key+' Avg:'+str(avgPrice))


      columnValueToScoreMapping=dict()


      for key in columnToAvgPriceMapping:

        columnValueToScoreMapping[key]=columnToAvgPriceMapping[key]/minimum

        print('Key:'+key+' Value:'+str(columnValueToScoreMapping[key]))


      scoredValues=[]

      for value in values:

        scoredValues.append(columnValueToScoreMapping[value])

      return scoredValues






