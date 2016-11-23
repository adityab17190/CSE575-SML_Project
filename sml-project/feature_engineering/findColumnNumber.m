function [ columnNumber ] = findColumnNumber( data, columnName )
%findColumnNumber Summary of this function goes here
%   Author = 'Tanmay Patil'

    headers = data(1,:);
%     IndexC = strfind(headers, columnName);
%     columnNumber = find(not(cellfun('isempty', IndexC)));
    IndexC = strcmpi(headers, columnName);
    columnNumber = find(IndexC);

end

