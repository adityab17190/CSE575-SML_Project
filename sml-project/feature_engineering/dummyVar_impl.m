function [ finalData ] = dummyVar_impl( data, columnNumber )
%UNTITLED Replace categorical variables with dummy variables
%   Author = 'Tanmay Patil'

    headers = data(1,:);
    data(1,:) = [];
    
    column = data(:,columnNumber);
    column = nominal(column);
    newColumns = dummyvar(column);
    
    [~, cols] = size(data);
    [~, newCols] = size(newColumns);
    newData = {};
    newHeaders = {};
    counter = 1;
    
    for i=1:cols
        if(i==columnNumber)
            currentColumnName = headers(i);
            for j=1:newCols
                newHeaders(counter) = strcat(currentColumnName, num2str(j));
                newData(:,counter) = num2cell(num2str(newColumns(:,j)));
                counter = counter + 1;
            end
        else
            newHeaders(counter) = headers(i);
            newData(:,counter) = data(:,i);
            counter = counter + 1;
        end
    end
    
    finalData = [newHeaders; newData];

end
