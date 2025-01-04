function [modes]=iceemdan(x,Nstd,NR,MaxIter,SNRFlag)
% The current is an improved version, introduced in:
 
%[1] Colominas MA, Schlotthauer G, Torres ME. "Improve complete ensemble EMD: A suitable tool for biomedical signal processing" 
%       Biomedical Signal Processing and Control vol. 14 pp. 19-29 (2014)
 
%The CEEMDAN algorithm was first introduced at ICASSP 2011, Prague, Czech Republic
 
%The authors will be thankful if the users of this code reference the work
%where the algorithm was first presented:
 
%[2] Torres ME, Colominas MA, Schlotthauer G, Flandrin P. "A Complete Ensemble Empirical Mode Decomposition with Adaptive Noise"
%       Proc. 36th Int. Conf. on Acoustics, Speech and Signa Processing ICASSP 2011 (May 22-27, Prague, Czech Republic)
 
%Author: Marcelo A. Colominas
%contact: macolominas@bioingenieria.edu.ar
%Last version: 25 feb 2015
desvio_x=std(x);
x=x/desvio_x;
[a,b]=size(x);
temp=zeros(b,1);
 modes=zeros(b,1);
 aux=zeros(a,b);
for i=1:NR
    white_noise{i}=randn(size(x));%creates the noise realizations
end
 
for i=1:NR
    modes_white_noise{i}=emd(white_noise{i},'display',0);%calculates the modes of white gaussian noise
end
% save interval modes_white_noise
for i=1:NR %calculates the first mode
    xi=x+Nstd*modes_white_noise{i}(:,1)'/std(modes_white_noise{i}(:,1));
    [temp, o, it]=emd(xi,'MaxNumIMF',1,'SiftMaxIterations',MaxIter,'display',0);
    aux=aux+(xi-temp')/NR;% nnnnnnnnnnnnnnnnJub局部包络
end
 
modes= (x-aux)'; %saves the first mode
medias = aux; %  r1
k=1;
aux=zeros(a,b);
es_imf = min(size(emd(medias(1,:),'SiftMaxIterations',MaxIter,'display',0)));
while es_imf>1 %calculates the rest of the modes
    for i=1:NR
        tamanio=size(modes_white_noise{i});
        if tamanio(2)>=k+1
            noise=modes_white_noise{i}(:,k+1);
            if SNRFlag == 2
                noise=noise/std(noise); %adjust the std of the noise
            end
            noise=Nstd*noise;
            try
                [temp,o,it]=emd(medias(1,:)+std(medias(1,:))*noise','MaxNumIMF',1,'SiftMaxIterations',MaxIter,'display',0);
            catch    
                temp=emd(medias(1,:)+std(medias(1,:))*noise','MaxNumIMF',1,'SiftMaxIterations',MaxIter,'display',0);
            end
        else
            try
                [temp, o, it]=emd(medias(1,:),'MaxNumIMF',1,'SiftMaxIterations',MaxIter,'display',0);
            catch
                temp=emd(medias(1,:),'MaxNumIMF',1,'SiftMaxIterations',MaxIter,'display',0);
            end
        end
        aux=aux+(medias(1,:)+std(medias(1,:))*noise'-temp')/NR;% r2 r3 r...
    end
    modes=[modes (medias(1,:)-aux)'];
    medias = aux;
    aux=zeros(size(x));
    k=k+1;
    es_imf = min(size(emd(medias(1,:),'SiftMaxIterations',MaxIter,'display',0)));
end
modes = [modes (medias(1,:))'];
modes=modes*desvio_x;