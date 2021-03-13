clc; clear all;
 
for jj=1:500 % averaging °¹¼ö
    fname=sprintf('%d.bin',jj-1);
%     fname = ['uwpi', num2str(jj), '.bin'];
    fid=fopen(fname, 'rb'); 
        Sig(:,jj)=fread(fid, 'int16');
    fclose(fid); 
end
 Data=Sig';
 h=1751;
 w=2001;
 Fs=10^6;
 %%
 for jjj=1:h
for jj=1:w
         Laserdiskv5(:,jj,h+1-jjj)=Data(:,w*(jjj-1)+jj);
end
 end  
 clear Data;
 for jjj=1:h
     for jj=1:w
         Laser1(:,jj,jjj)=ConverSion(Laserdiskv5(:,jj,jjj));
%          Laser7v(:,jj,jjj)=(Laser7d(2:end,jj,jjj)-Laser7d(1:end-1,jj,jjj)).*Fs;
%          Laser7a(:,jj,jjj)=(Laser7v(2:end,jj,jjj)-Laser7v(1:end-1,jj,jjj)).*Fs;
     end
 end

save('Laser1.mat','Laser1');
% save('Laser7v.mat','Laser7v');
% save('Laser7a.mat','Laser7a');
%%
% SRS(:,1)=srs7(10^6,Laser7a(:,148,60));
% SRS(:,2)=srs7(10^6,Laser7a(:,63,60));
% [SRS(:,3),fn]=srs7(10^6,Laser7a(:,63,15));
% plot(fn,SRS)
% set(gca,'MinorGridLineStyle',':','GridLineStyle',':','XScale','log','YScale','log','fontsize',25,'FontWeight','b','linewidth',0.5);
%%