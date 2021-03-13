%convert UWPI bin to txt
% This code will read binary file and convert into 
% text format for xls  
close all; clear all;
    clc;
x = 400/0.2+1; %number of points in one column
y = 350/0.2+1; %number of points in one row

samples = 5; %number of samples or number of files

b = ones(y,x); % scan area area matrix (x,y) 

 for i = 0 : samples-1 %number of files

fname = sprintf('%d.bin', i);
fname2 = sprintf('CSV2/UWPI%d.csv', i);

fid=fopen(fname,'rb'); 
fid2=fopen(fname2,'w'); 
a = fread(fid, 'int16');

    for k= 0: y-1 
    b(k+1,:) = a(k*x+1 : (k+1)*x);
    end;

    i
dlmwrite(fname2, b, '-append');
fclose (fid); 
fclose (fid2); 

 end;
  display('finish');