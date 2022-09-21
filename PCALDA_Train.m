%function [FFACE,TotalMeanFACE,pcaTotalFACE,projectPCA,eigvector,prototypeFACE]=PCALDA_Train

function [FFACE, pcaTotalFACE, projectPCA, prototype, testcaseFACE, rightNum, rightPer]=PCALDA_Train

people=40;%40組資料

withinsample=5;%5組資料為SAMPLE

principlenum=50;%50維

FFACE=[];%1024維*200筆資料，存原始資料

ldanum = 50;
totalcount = 0;
answer = 0;

for k=1:1: people
    for m=1:2:10
        matchstring=['orl3232' '\' num2str(k) '\' num2str(m) '.bmp'];
        matchX=imread(matchstring);
        matchX=double(matchX);      %bmp->double
        if(k==1 && m==1)
            [row,col]=size(matchX);
        end

        matchtempF=[];
        for n=1:row
            matchtempF=[matchtempF,matchX(n,:)];%把一個圖檔變成一個vector
        end

        FFACE=[FFACE; matchtempF];%把每一個vector一個一個存進去
    end
end

TotalMeanFACE=mean(FFACE);   %1*1024算術平均數
zeromeanTotalFACE=FFACE;

for i=1:1:withinsample*people
    for j=1:1:(row)*(col)
        zeromeanTotalFACE(i,j)=zeromeanTotalFACE(i,j)-TotalMeanFACE(j); %正規化
    end
end

SST=zeromeanTotalFACE'*zeromeanTotalFACE; %共變異數矩陣1024*1024

[Vec,Val] = eig(SST);

eigvalue=diag(Val);

[junk,index] = sort(eigvalue,'descend');

PCA = Vec(:,index);
eigvalue=eigvalue(index);

projectPCA=PCA(:,1:principlenum); %newdata取前面50項1024*50
pcaTotalFACE=[];

for i=1:1:withinsample*people
    tempFACE=zeromeanTotalFACE(i,:);
    tempFACE=tempFACE*projectPCA;         % 內積求新座標值
    pcaTotalFACE=[pcaTotalFACE;tempFACE]; %儲存所有投影至PCA空間中的訓練影像
end

%------- PCA transform ------------------------------------------------------------------------------
for i=1:5:5*40  %暫存單依類別PCA空間中訓練影像                                                                                                                                                                                                                                                                                                                                                                                                                                    
    within=pcaTotalFACE(i:i+5-1,:);
    if(i==1)
        meanwithinFACE=mean(within)
        within=within-meanwithinFACE
        SW=within'*within; %組內變異
        ClassMean=mean(within);
    end
    if(i>1)
        meanwithinFACE=mean(within)
        within=within-meanwithinFACE
        SW=SW+within'*within; 
        ClassMean=[ClassMean;mean(within)];%this amtrix is for between
    end
end
pcatotalmean=mean(pcaTotalFACE)
SB=ClassMean'*ClassMean;
[eigvector,eigvalue] = eig(inv(SW)*SB);
eigvalue=diag(eigvalue);
[junk,index]=sort(eigvalue,'descend');
eigvalue=eigvalue(index);
projectLDA=eigvector(:,index);

projectLDA=projectLDA(:,1:30);    %取前面30項50*30
prototype=pcaTotalFACE*projectLDA;
%++++++++ LDA transform ++++++++++++++++++++++++++++++++
FFACE = [];
for k=1:1: people
     % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     for m=2:2:10
         matchstring=['orl3232' '\' num2str(k) '\' num2str(m) '.bmp'];
         matchX=imread(matchstring);
         matchX=double(matchX);                    % bmp -> double
         if (k==1 && m==2)
            [row,col]=size(matchX);
         end
         
         matchtempF=[];
         % --arrange the image into a vector
         for n=1:row
             matchtempF=[matchtempF, matchX(n,:)]; % Matrix 1 * 1024 == a picture
         end
         
         FFACE=[FFACE; matchtempF]; % FFACE is the data of all pic
     end 
     % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
end % end of k=1:1:people

TotalMeanFACE=mean(FFACE);      % 1024 維度 => Matrix 1 * 1024
zeromeanTotalFACE=FFACE;

%++++++++++ zero mean ++++++++++++++++++++++++++++++++
for i=1:1:withinsample*people
    for j=1:1:(row)*(col) 
        zeromeanTotalFACE(i,j)=zeromeanTotalFACE(i,j) - TotalMeanFACE(j); % test case 正規化
    end
end

pcaTotalFACE = [];
for i=1:1:withinsample*people
    tempFACE=zeromeanTotalFACE(i,:);
    tempFACE=tempFACE*projectPCA;                  % 內積求新座標值
    pcaTotalFACE=[pcaTotalFACE;tempFACE];               %儲存所有test case投影至PCA空間中的訓練影像
end

testcaseFACE=pcaTotalFACE*eigvector(:,1:30);
ans = 0;
rightNum = 0;
for i=1:1:200       % test case
    eucdis = 0;     
    for j=1:1:200   % prototype case
        oaf = testcaseFACE(i,:) - prototype(j,:);
        eucdisTemp = oaf*oaf';
        if (j==1 || eucdis>eucdisTemp)
            eucdis = eucdisTemp;
            ans = j;
            oafTemp = oaf;
        end
    end

    if (floor((ans-1) / 5) * 5 + 1 <= i && i <= (floor((ans-1) / 5) + 1) * 5)
        rightNum = rightNum + 1;
    end
end

rightPer = rightNum / 200;

end