function [ ] = writeToFile( data, outputFileName )
%writeToFile Summary of this function goes here
%   Author = 'Tanmay Patil'

    fid=fopen(outputFileName,'wt');
    [rows, ~] = size(data);
    for i=1:rows
        fprintf(fid,'%s,',data{i,1:end-1});
        fprintf(fid,'%s\n',data{i,end});
    end
    fclose(fid);

end

