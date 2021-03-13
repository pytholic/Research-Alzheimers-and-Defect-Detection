close all; clear all;
    clc;

x = 400/0.2+1; %number of points in one column
y = 350/0.2+1; %number of points in one row

samples = 500;

%n=1;


%for k = 167:178
    %for j = 291:302 
for k = 362:366
    for j = 153:162
        for i = 0 : samples-1
    
        fname = sprintf('%d.bin', i);
        fid=fopen(fname,'rb');
        
        Data = fread(fid, 'int16');
        
        %Data = dlmread(fname);
        value = Data(j);

        display(value);
%fname = (point_1.xls);
%fid=fopen(fname,'w');
%display (value);
        fname2 = sprintf('1D_new/defected/point(%d,%d).csv',j,k);  
        %a = value;
        dlmwrite(fname2, value , '-append');
        %dlmwrite(fname2, value , '\t');
        %dlmwrite(fname2,' ', '-append', 'coffset', n);
        fclose (fid);
        end;
    %dlmwrite(fname2,' ', '-append', 'coffset', n);  
    %n=n+1
    end;

end; 
display('finish');
