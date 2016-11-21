function [ finalData, rowsTrain, rowsTest ] = mergeData( file1, file2 )
%mergeData Summary of this function goes here
%   Author = 'Tanmay Patil'

    dataTrain = read_mixed_csv(file1, ',');
    headersTrain = dataTrain(1,:);
    dataTrain(1,:) = [];
    [rowsTrain colsTrain] = size(dataTrain);
    
    dataTest = read_mixed_csv(file2, ',');
    dataTest(1,:) = [];
    [rowsTest colsTest] = size(dataTest);
    
    finalData = [headersTrain; dataTrain; dataTest];

end

