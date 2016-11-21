function [ newX ] = replaceVariables( X, replace_X, unique_X )
%replace_with_categorical_vars Replace variables using this method
%   Author = 'Tanmay Patil'

    newX={};
    
    for i=1:length(X)
        val = X(i);
        index = find(unique_X == val);
        newX(i,:) = replace_X(index);
    end

end

