close all; clear all;
    clc;

y = 400/0.2+1; %number of points in one column
x = 350/0.2+1; %number of points in one row

samples = 500;

n = 1;
for k = 1 : x
    for j = 1 : y
        for i = 0 : samples-1
    
        fname = sprintf('excel/UWPI%d.xls', i);    
        Data = dlmread(fname);
        value = Data(k,j);

        display(value);
%fname = (point_1.xls);
%fid=fopen(fname,'w');
%display (value);
        fname2 = sprintf('1D/point_%d.xls', n); 
        dlmwrite(fname2, value, '-append');
        end;
    n = n+1;    
    end;
end; 
display('finish');
