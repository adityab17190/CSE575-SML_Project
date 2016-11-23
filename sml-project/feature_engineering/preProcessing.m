function [ ] = preProcessing()
%preProcessing Summary of this function goes here
%   Author = 'Tanmay Patil'

    inputFileNameTrain = 'data/aditya_filtered_train.csv';
    inputFileNameTest = 'data/aditya_filtered_test.csv';
    
    [data, rowsTrain, rowsTest] = mergeData(inputFileNameTrain, inputFileNameTest);
    
    removeColumnsArray = {'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','HouseStyle','Condition1','Condition2','LotFrontage','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','BldgType','CentralAir','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'};
    
    for t=1:length(removeColumnsArray)
        currentVar = removeColumnsArray(t);
        cColumnNumber = findColumnNumber(data, currentVar);
        data = removeColumns(data, cColumnNumber);
    end
    
    unique_a = [20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190];
    replace_a = {'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16'};
    aColumnNumber = findColumnNumber(data, 'MSSubClass');
    data = convertToCategoricalVars(data, unique_a, replace_a, aColumnNumber);
    
    unique_b = [10,9,8,7,6,5,4,3,2,1];
    replace_b = {'B1','B2','B3','B4','B5','B6','B7','B8','B9','B10'};
    bColumnNumber = findColumnNumber(data, 'OverallQual');
    data = convertToCategoricalVars(data, unique_b, replace_b, bColumnNumber);
    
    unique_c = [10,9,8,7,6,5,4,3,2,1];
    replace_c = {'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10'};
    cColumnNumber = findColumnNumber(data, 'OverallCond');
    data = convertToCategoricalVars(data, unique_c, replace_c, cColumnNumber);
    
    allDummyVars = {'MSSubClass', 'Neighborhood', 'MSZoning', 'Street', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'Heating', 'HeatingQC'};
    
    for t=1:length(allDummyVars)
        currentVar = allDummyVars(t);
        disp('------');
        disp(currentVar);
        disp('------');
        bColumnNumber = findColumnNumber(data, currentVar);
        data = dummyVar_impl(data, bColumnNumber);
    end
    
    [trainData, testData] = splitData(data, rowsTrain, rowsTest);

    
    writeToFile(trainData, 'data/filtered_train_new_scored.csv');
    writeToFile(testData, 'data/filtered_test_new_scored.csv');

end

