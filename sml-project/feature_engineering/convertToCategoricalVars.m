function [ data ] = convertToCategoricalVars( data, unique_a, replace_a, aColumnNumber )
%UNTITLED2 Summary of this function goes here
%   Author = 'Tanmay Patil'

    headers = data(1,:);
    data(1,:) = [];
    
    column = data(:,aColumnNumber);
    column = str2double(column);
    newA = replaceVariables(column, replace_a, unique_a);
    data(:,aColumnNumber) = newA;
    
    data = [headers; data];
end

