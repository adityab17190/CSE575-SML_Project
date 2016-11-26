function [ data ] = processSkewness( data )
%processSkewness
%   Author = 'Tanmay Patil'

    columnNumber = findColumnNumber(data, 'SalePrice');
    
    headers = data(1,:);
    data(1,:) = [];
    
    data = str2double(data);
    a = data(:,columnNumber);
    a = log1p(a);
    data(:,columnNumber) = a;
    
    ids = data(:,1);
    data(:,1) = [];
    
    [~, p] = size(data);

    for i=1:p
        currentCol = data(:,i);
        currSkewness = skewness(currentCol);
        if(currSkewness > 0.65)
            currentCol = log1p(currentCol);
            data(:,i) = currentCol;
        end
    end
    
    data = [ids data];
    
    tmp = num2cell(data);
    data = cellfun(@(x)sprintf('%i',x),tmp,'uni',false);
    
    data = [headers; data];

end

