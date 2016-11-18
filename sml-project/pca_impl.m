function [ ] = pca_impl( inputFileName, outputFileName )
%pca_impl Reduce Dimensionality of the input file
%   Author: Tanmay Patil

    originalX = csvread(inputFileName, 1, 0);
    X = originalX;
    [originalM, originalP] = size(originalX); % m data points in p-dimensional space
    
    % Remove 1st column(Index) and last column(Y)
    X(:,originalP) = [];
    X(:,1) = [];
    
%     normX = normc(X);
    
    [coeff, score] = pca(X);
    
    reducedMatrix = X*coeff;
    newData = [originalX(:,1) reducedMatrix];
    
    [reducedM, reducedP] = size(newData); % reduced m & p
    newData = [1:reducedP; newData];
    
    csvwrite(outputFileName,newData);

end

