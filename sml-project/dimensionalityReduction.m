function [ ] = dimensionalityReduction( )
% dimensionalityReduction Reduce Dimensionality of the input file
%   Author: Tanmay Patil

    inputFileNameTraining = 'data/feature_engineering_test/train_new_scored.csv';
    outputFileNameTraining = 'data/feature_engineering_test/PCA_train_new_scored.csv';
    
    inputFileNameTest = 'data/feature_engineering_test/test_new_scored.csv';
    outputFileNameTest = 'data/feature_engineering_test/PCA_test_new_scored.csv';
    
    pca_impl(inputFileNameTraining, outputFileNameTraining);
    pca_impl(inputFileNameTest, outputFileNameTest);

end
