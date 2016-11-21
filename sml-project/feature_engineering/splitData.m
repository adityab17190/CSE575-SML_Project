function [ trainData, testData ] = splitData( data, rowsTrain, rowsTest )
%splitData Summary of this function goes here
%   Author = 'Tanmay Patil'

    headers = data(1,:);
    data(1,:) = [];
    
    trainData = data(1:rowsTrain,:);
    testData = data(rowsTrain+1:rowsTrain+rowsTest,:);
    
    trainData = [headers; trainData];
    testData = [headers; testData];

end

