function [ columnNumber ] = findColumnNumber( data, columnName )
%findColumnNumber Summary of this function goes here
%   Author = 'Tanmay Patil'

    headers = data(1,:);
    IndexC = strfind(headers, columnName);
    columnNumber = find(not(cellfun('isempty', IndexC)));

end

