%convert UWPI bin to txt
% This code will read binary file and convert into 
% text format for xls  
close all; clear all;
    clc;
y = 400/0.2+1; %number of points in one column
x = 350/0.2+1; %number of points in one row

samples = 500; %number of samples or number of files

b = ones(x,y); % scan area area matrix (x,y) 

 for i = 0 : samples-1 %number of files

fname = sprintf('%d.bin', i);
fname2 = sprintf('CSV/UWPI%d.csv', i);

fid=fopen(fname,'rb'); 
fid2=fopen(fname2,'w'); 
a = fread(fid, 'int16');

    for k= 0: x-1 
    
    %b(k+1,:) = a(k*y+1 : (k+1)*y);
    end;

    i
dlmwrite(fname2, b, '-append');
fclose (fid); 
fclose (fid2); 

 end;
  display('finish');