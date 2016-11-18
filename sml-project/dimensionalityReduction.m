function [ ] = dimensionalityReduction( )
% dimensionalityReduction Reduce Dimensionality of the input file
%   Author: Tanmay Patil

    inputFileNameTraining = 'data/train_scored.csv';
    outputFileNameTraining = 'data/PCA_train_scored.csv';
    
    inputFileNameTest = 'data/test_scored.csv';
    outputFileNameTest = 'data/PCA_test_scored.csv';
    
    pca_impl(inputFileNameTraining, outputFileNameTraining);
    pca_impl(inputFileNameTest, outputFileNameTest);

end
