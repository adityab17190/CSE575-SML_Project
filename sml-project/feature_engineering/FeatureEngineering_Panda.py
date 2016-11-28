__author__ = 'aditya'


import pandas as pd
import numpy as np


trainDataFile='../data/train.csv'
testDataFile='../data/test.csv'


def transformFeatures():

    train_df=pd.read_csv(trainDataFile)
    test_df=pd.read_csv(testDataFile)

    test_df.loc[666, "GarageQual"] = "TA"
    test_df.loc[666, "GarageCond"] = "TA"
    test_df.loc[666, "GarageFinish"] = "Unf"
    test_df.loc[666, "GarageYrBlt"] = "1980"

    test_df.loc[1116, "GarageType"] = np.nan

    performMunging(train_df,'../data/processed_train.csv', True)
    performMunging(test_df,'../data/processed_test.csv', False)




def performMunging(df,filePath,isTrainData):

    all_df=pd.DataFrame(index=df.index)

    all_df['Id']=df['Id']

    #all_df["Street"]=(df["Street"]=="Pave")*1

    #all_df['Alley']=(df['Alley']!='NA')*1

    all_df['LotShape']=(df['LotShape']=='Reg')*1

    #all_df['LandContour']=(df['LandContour']=='Lvl')*1

    #all_df['Utilities']=(df['Utilities']=='AllPub') *1

    all_df['LotConfig']=(df['LotConfig']=='Inside')*1

    #all_df['LanSlope']=(df['LandSlope']=='Gtl')*1

    all_df['1stFlrSF']=df['1stFlrSF']

    all_df['2ndFlrSF']=df['2ndFlrSF']

    all_df['GarageArea']=df['GarageArea']

    all_df['GrLivArea']=df['GrLivArea']

    area_columns = ['LotFrontage', 'LotArea', 'LowQualFinSF','MasVnrArea','PoolArea','3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch' ]

    all_df['TotalArea']=df[area_columns].sum(axis=1)

    #print(all_df['TotalArea'])

    all_df['BedroomAbvGr']=df['BedroomAbvGr']

    neighborhood_map={

        'MeadowV':     0,

        'IDOTRR':     1,

        'BrDale':     1,

        'BrkSide':    1,

        'Edwards':    1,

        'OldTown':    1,

        'Sawyer':     1,

        'Blueste':    1,

        'SWISU':      1,

        'NPkVill':    1,

        'NAmes' :     1,

        'Mitchel':    1,

        'SawyerW':    2,

        'NWAmes':     2,

        'Gilbert':    2,

        'Blmngtn':    2,

        'CollgCr':    2,

        'Crawfor':    3,

        'ClearCr':    3,

        'Somerst':    3,

        'Veenker':    3,

        'Timber':     3,

        'StoneBr':    4,

        'NridgHt':    4,

        'NoRidge':    4

    }



    all_df['Neighborhood']=df['Neighborhood'].map(neighborhood_map)

    all_df['Age']=2016-df['YearRemodAdd']

    all_df['TimeSinceSold']=2016-df['YrSold']+(8-df['MoSold'])/12

    df['GarageYrBlt'].replace('NaN',df['YearBuilt'],inplace='True')

    print('Garage Year Built')

    print(df['GarageYrBlt'])

    all_df['GarageAge']=2016-df['GarageYrBlt'].apply(int)

    print(all_df['GarageAge'])

    qualityValuesMap={

        None: 0,
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5


    }

    exterMap={

        None: 0,
        "Po": 1,
        "Fa": 1,
        "TA": 1,
        "Gd": 2,
        "Ex": 2

    }

    bsmtMap={

        None: 0,
        "Po": 1,
        "Fa": 1,
        "TA": 1,
        "Gd": 2,
        "Ex": 2

    }




    all_df['ExterQual']=df['ExterQual'].map(exterMap)

    all_df['ExterCond']=df['ExterCond'].map(exterMap)

    all_df['BsmtQual']=df['BsmtQual'].map(bsmtMap)

    all_df['BsmtCond']=df['BsmtCond'].map(qualityValuesMap)

    all_df['HeatingQC']=df['HeatingQC'].map(qualityValuesMap)

    all_df['KitchenQual']=df['KitchenQual'].map(qualityValuesMap)

    all_df['FireplaceQu']=df['FireplaceQu'].map(qualityValuesMap)

    all_df['GarageQual']=df['GarageQual'].map(qualityValuesMap)

    all_df['GarageCond']=df['GarageCond'].map(qualityValuesMap)


    #print(all_df['HeatingQC'])

    #print(all_df['GarageCond'])

    #print(df['SalePrice'].groupby(df['MSZoning']).median().order())

    msZoningMap={
        None:0,
        'C (all)':1,
        'RM':2,
        'RH':2,
        'RL':3,
        'FV':4
    }

    all_df['MSZoning']=df['MSZoning'].map(msZoningMap)

    msSubCLassMap={
        180:1,
        30:1,
        45:1,
        190:2,
        50:2,
        90:2,
        85:2,
        40:2,
        160:3,
        70:3,
        20:3,
        75:3,
        80:3,
        120:4,
        60:4

    }

    print(df['SalePrice'].groupby(df['MSSubClass']).median().order())

    all_df['MSSubClass']=df['MSSubClass'].map(msSubCLassMap)

    if not isTrainData:
        all_df['MSSubClass'][1358]='0'

    #all_df['Condition1']=(df['Condition1']=='Norm')*1

    #all_df['Condition2']=(df['Condition2']=='Norm')*1



    #print(all_df['Condition2'])

    print(df['SalePrice'].groupby(df['BldgType']).median().order())



    bldgTypeMap={

        '2fmCon':1,
        'Duplex':2,
        'Twnhs':2,
        '1Fam':3,
        'TwnhsE':3
        }

    all_df['BldgType']=df['BldgType'].map(bldgTypeMap)

    print(all_df['BldgType'])

    print(df['SalePrice'].groupby(df['HouseStyle']).median().order())

    houseStyleMap={

        '1.5Unf':1,
        '1.5Fin':2,
        '2.5Unf':2,
        'SFoyer':2,
        '1Story':3,
        'SLvl':3,
        '2Story':4,
        '2.5Fin':4
    }

    all_df['HouseStyle']=df['HouseStyle'].map(houseStyleMap)

    #print(all_df['HouseStyle'])

    all_df['MasVnrType']=(df['MasVnrType']=='BrkFace')*1

    print(all_df['MasVnrType'])

    all_df['Exterior1st_VinylSd']=(df['Exterior1st']=='VinylSd')*1

    all_df['Exterior1st_HdBoard']=(df['Exterior1st']=='HdBoard')*1

    all_df['Exterior1st_MetalSd']=(df['Exterior1st']=='MetalSd')*1

    all_df['Exterior1st_Wd Sdng']=(df['Exterior1st']=='Wd Sdng')*1

    all_df['Exterior2nd_VinylSd']=(df['Exterior2nd']=='VinylSd')*1

    all_df['Exterior2nd_HdBoard']=(df['Exterior2nd']=='HdBoard')*1

    all_df['Exterior2nd_MetalSd']=(df['Exterior2nd']=='MetalSd')*1

    all_df['Exterior2nd_Wd Sdng']=(df['Exterior2nd']=='Wd Sdng')*1

    #print(all_df['Exterior1st_HdBoard'])

    all_df['Foundation_CBlock']=(df['Foundation']=='CBlock')*1

    all_df['Foundation_PConc']=(df['Foundation']=='PConc')*1

    #print(all_df['Foundation_CBlock'])

    #all_df['Heating']=(df['Heating']=='GasA')*1

    all_df['CentralAir']=(df['CentralAir']=='Y')*1

    #all_df['Electrical']=(df['Electrical']=='SBrKr')*1

    all_df['BsmtFullBath']=df['BsmtFullBath']

    all_df['BsmtHalfBath']=df['BsmtHalfBath']

    all_df['FullBath']=df['FullBath']

    all_df['HalfBath']=df['HalfBath']

    all_df['BedroomAbvGr']=df['BedroomAbvGr']

    all_df['KitchenAbvGr']=df['KitchenAbvGr']

    all_df['TotRmsAbvGrd']=df['TotRmsAbvGrd']

    all_df['Functional']=(df['Functional']=='Typ')*1

    all_df['Fireplaces']=df['Fireplaces']

    all_df['GarageType_Attached']=(df['GarageType']=='Attchd')*1

    all_df['GarageType_Detached']=(df['GarageType']=='Detchd')*1

    all_df['GarageType_BuiltIn']=(df['GarageType']=='BuiltIn')*1

    all_df['GarageFinish_Unf']=(df['GarageFinish']!='Unf')*1

    all_df['GarageFinish_RFn']=(df['GarageFinish']!='RFn')*1

    all_df['GarageFinish_Fin']=(df['GarageFinish']!='Fin')*1

    all_df['GarageCars']=df['GarageCars']

    #all_df['PavedDrive']=(df['PavedDrive']=='Y')*1

    all_df['Fence']=(df['Fence']!=None)*1

    #all_df['MiscFeature']=(df['MiscFeature']=='Shed')*1

    all_df['MiscVal']=df['MiscVal']

    #all_df['SaleType']=(df['SaleType']=='WD')*1

    #all_df['SaleCondition']=(df['SaleCondition']=='Normal')*1

    if isTrainData:

        all_df['SalePrice']=df['SalePrice']

    all_df.to_csv(filePath,index=False)


if __name__ == '__main__':
    transformFeatures()