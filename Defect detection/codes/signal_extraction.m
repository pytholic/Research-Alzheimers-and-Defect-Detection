close all; clear all;
    clc;

%y = 400/0.2+1; %number of points in one column
%x = 350/0.2+1; %number of points in one row

samples = 500;
for i = 0 : samples-1

fname = sprintf('excel/UWPI%d.xls', i);    
Data = dlmread(fname);
value = Data(1,1);

display(value);
%fname = (point_1.xls);
%fid=fopen(fname,'w');
%display (value);
%fname2 = sprintf('1D/point_%d.xls', i); 
dlmwrite('1D/point_1_1.csv', value, '-append'); 

 end;
 display('finish');
