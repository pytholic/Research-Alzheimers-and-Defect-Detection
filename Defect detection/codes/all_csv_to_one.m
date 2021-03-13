close all; clear all;
    clc;

y = 400/0.2+1; %number of points in one column
x = 350/0.2+1; %number of points in one row

samples = 500;

n = 1;
for k = 1 : x
    for j = 1 : y
        for i = 0 : samples-1
    
        fname = sprintf('1D/point_%d.csv', n);    
        Data = csvread(fname);
        value = Data(j);

        display(value);
%fname = (point_1.xls);
%fid=fopen(fname,'w');
%display (value);
        fname2 = sprintf('1D_2/point_all2.csv'); 
        csvwrite(fname2, value, k, j);
        end;
    n = n+1;    
    end;
end; 
display('finish');
