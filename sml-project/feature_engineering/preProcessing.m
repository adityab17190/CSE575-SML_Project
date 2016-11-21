function [ ] = preProcessing()
%preProcessing Summary of this function goes here
%   Author = 'Tanmay Patil'

    inputFileNameTrain = 'data/train.csv';
    inputFileNameTest = 'data/test.csv';
    
    [data, rowsTrain, rowsTest] = mergeData(inputFileNameTrain, inputFileNameTest);
    
    unique_a = [20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190];
    replace_a = {'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16'};
    aColumnNumber = findColumnNumber(data, 'MSSubClass');
    data = convertToCategoricalVars(data, unique_a, replace_a, aColumnNumber);
    data = dummyVar_impl(data, aColumnNumber);
    
    bColumnNumber = findColumnNumber(data, 'Neighborhood');
    data = dummyVar_impl(data, bColumnNumber);
    
    [trainData, testData] = splitData(data, rowsTrain, rowsTest);
    
    writeToFile(trainData, 'data/train_new_scored.csv');
    writeToFile(testData, 'data/test_new_scored.csv');

end

