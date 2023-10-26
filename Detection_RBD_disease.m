
#data obtained from CAP Sleep Database

load("rbd1.mat"); f1=rbd1;
load("rbd2.mat"); f2=rbd2;
load("rbd3.mat"); f3=rbd3;
load("rbd4.mat"); f4=rbd4;
load("rbd5.mat"); f5=rbd5;
load("rbd6.mat"); f6=rbd6;
load("rbd7.mat"); f7=rbd7;
load("rbd8.mat"); f8=rbd8;
load("rbd9.mat"); f9=rbd9;
load("rbd10.mat"); f10=rbd10;
load("rbd11.mat"); f11=rbd11;
load("rbd12.mat"); f12=rbd12;
load("rbd13.mat"); f13=rbd13;
load("rbd14.mat"); f14=rbd14;
load("rbd15.mat"); f15=rbd15;
load("rbd16.mat"); f16=rbd16;
load("rbd17.mat"); f17=rbd17;
load("rbd18.mat"); f18=rbd18;
load("rbd19.mat"); f19=rbd19;
load("rbd20.mat"); f20=rbd20;
load("rbd21.mat"); f21=rbd21;
load("rbd22.mat"); f22=rbd22;



for i=1:1:22

switch i
    case 1
        N=rbd1;
    case 2
        N=rbd2;
    case 3
        N=rbd3;
    case 4
        N=rbd4;
    case 5
        N=rbd5;
    case 6
        N=rbd6;
    case 7
        N=rbd7;
    case 8
        N=rbd8;
    case 9
        N=rbd9;
    case 10
        N=rbd10;
    case 11
        N=rbd11;
    case 12
        N=rbd12;
    case 13
        N=rbd13; 
    case 14
        N=rbd14;
    case 15
        N=rbd15;
    case 16
        N=rbd16;
    case 17
        N=rbd17;
    case 18
        N=rbd18;
    case 19
        N=rbd19;
    case 20
        N=rbd20;
    case 21
        N=rbd21;
    case 22
        N=rbd22;        
    otherwise
        disp('hata')
end



N=normalize(N)

%Divided Epoch

L=length(N)
fs =15360

maxepoch=L/fs;
for h=1:maxepoch
 if h==1
    epoch(:,h) = N(1:fs);
 else 
    epoch(:, h) = N((h-1)*fs+1:(h*fs));
 end
end




for o=1:maxepoch
%Preprossing
EP=epoch(:,o);

thr=(2/4)*max(EP);
for j=1:length(EP)
 if EP(j)>thr
 y1(j,1)=mean(EP);
 else if EP(j)<-thr
 y1(j,1)=-mean(EP);
 else
 y1(j,1)=EP(j);
 end
 end
end
s = bandpass(y1,[0.5 14],200);
%Features
zc=0;
l=length(s);
for k=1:length(s)-1
 if s(k)*s(k+1)<0
 zc=zc+1; 
 end
end
feature_epoch(o,1)=zc;
varyans=var(s);
feature_epoch(o,2)=varyans;

medianFreq=medfreq(s);
feature_epoch(o,3)=medianFreq;

kur = kurtosis(s);
feature_epoch(o,4)=kur;

skew = skewness(s);
feature_epoch(o,5)=skew;

m= mean(s);
feature_epoch(o,6)=m;

stde= std(s);
feature_epoch(o,7)=stde;

rmdd = rms(s);
feature_epoch(o,8)=rmdd;

meanab= mad(s);
feature_epoch(o,9)=meanab;

inte = iqr(s);
feature_epoch(o,10)=inte;

entr = entropy(s);
feature_epoch(o,11)=entr;

Nx = length(s);
nsc = floor(Nx/4.5);
nov = floor(nsc/2);
nff = max(256,2^nextpow2(nsc));

t = pwelch(s,hamming(nsc),nov,nff);
feature_epoch(o,12)=sum(t);

Mobility=sqrt(var(diff(s))/var(s));
Complexity=sqrt(var(diff(diff(s)))/var(s))/sqrt(var(diff(s))/var(s));

feature_epoch(o,13)=Mobility;
feature_epoch(o,14)=Complexity;

pbandDelta = bandpower(s,100,[0.4 4]);
pbandTheta = bandpower(s,100,[4 8]);
pbandAlfa = bandpower(s,100,[8 13]);

feature_epoch(o,15)=pbandDelta;
feature_epoch(o,16)=pbandTheta;
feature_epoch(o,17)=pbandAlfa;

feature_epoch(o,18)=max(s)-min(s);

%MEAN FEATURE


for w=1:18

feature(1,w)=sum(feature_epoch(:,w))/length(feature_epoch(:,1));

end

end
switch i
    case 1
       featurerbd=feature;
save('featurerbd1.mat','featurerbd');
    case 2
       featurerbd=feature;
save('featurerbd2.mat','featurerbd');
        
    case 3
       featurerbd=feature;
save('featurerbd3.mat','featurerbd');
    case 4
       featurerbd=feature;
save('featurerbd4.mat','featurerbd');
    case 5
       featurerbd=feature;
save('featurerbd5.mat','featurerbd');
    case 6
       featurerbd=feature;
save('featurerbd6.mat','featurerbd');
    case 7
       featurerbd=feature;
save('featurerbd7.mat','featurerbd');
    case 8
       featurerbd=feature;
save('featurerbd8.mat','featurerbd');
    case 9
       featurerbd=feature;
save('featurerbd9.mat','featurerbd');
    case 10
       featurerbd=feature;
save('featurerbd10.mat','featurerbd');
    case 11
       featurerbd=feature;
save('featurerbd11.mat','featurerbd');
    case 12
       featurerbd=feature;
save('featurerbd12.mat','featurerbd');
    case 13
       featurerbd=feature;
save('featurerbd13.mat','featurerbd');
    case 14
       featurerbd=feature;
save('featurerbd14.mat','featurerbd');
    case 15
       featurerbd=feature;
save('featurerbd15.mat','featurerbd');
    case 16
       featurerbd=feature;
save('featurerbd16.mat','featurerbd');
    case 17
       featurerbd=feature;
save('featurerbd17.mat','featurerbd');
    case 18
       featurerbd=feature;
save('featurerbd18.mat','featurerbd');
    case 19
       featurerbd=feature;
save('featurerbd19.mat','featurerbd');
    case 20
       featurerbd=feature;
save('featurerbd20.mat','featurerbd');
    case 21
       featurerbd=feature;
save('featurerbd21.mat','featurerbd');
    case 22
       featurerbd=feature;
save('featurerbd22.mat','featurerbd');
    otherwise
        disp('hata')
end

 feature=[];
 feature_epoch=[];
 j=0;
 k=0;
end
clc

load("n1.mat"); f1=n1;
load("n2.mat"); f2=n2;
load("n3.mat"); f3=n3;
load("n4.mat"); f4=n4;
load("n5.mat"); f5=n5;
load("n6.mat"); f6=n6;
load("n7.mat"); f7=n7;
load("n8.mat"); f8=n8;
load("n9.mat"); f9=n9;
load("n10.mat"); f10=n10;
load("n11.mat"); f11=n11;
load("n12.mat"); f12=n12;
load("n13.mat"); f13=n13;
load("n14.mat"); f14=n14;
load("n15.mat"); f15=n15;
load("n16.mat"); f16=n16;


for i=1:1:16

switch i
    case 1
        N=n1;
    case 2
        N=n2;
    case 3
        N=n3;
    case 4
        N=n4;
    case 5
        N=n5;
    case 6
        N=n6;
    case 7
        N=n7;
    case 8
        N=n8;
    case 9
        N=n9;
    case 10
        N=n10;
    case 11
        N=n11;
    case 12
        N=n12;
    case 13
        N=n13; 
    case 14
        N=n14;
    case 15
        N=n15;
    case 16
        N=n16;       
    otherwise
        disp('hata')
end







L=length(N);


% 15361 =30 sn
%5634 variable = 60
%2815 =30 sn
fs =15360;

maxepoch=L/fs;
for h=1:maxepoch
 if h==1
    epoch(:,h) = N(1:fs);
 else 
    epoch(:, h) = N((h-1)*fs+1:(h*fs));
 end
end




for o=1:maxepoch
%Preprossing
EP=epoch(:,o);

thr=(2/4)*max(EP);
for j=1:length(EP)
 if EP(j)>thr
 y1(j,1)=mean(EP);
 else if EP(j)<-thr
 y1(j,1)=-mean(EP);
 else
 y1(j,1)=EP(j);
 end
 end
end
s = bandpass(y1,[0.5 14],200);
%features
zc=0;
l=length(s);
for k=1:length(s)-1
 if s(k)*s(k+1)<0
 zc=zc+1; 
 end
end
feature_epoch(o,1)=zc;
varyans=var(s);
feature_epoch(o,2)=varyans;

medianFreq=medfreq(s);
feature_epoch(o,3)=medianFreq;

kur = kurtosis(s);
feature_epoch(o,4)=kur;

skew = skewness(s);
feature_epoch(o,5)=skew;

m= mean(s);
feature_epoch(o,6)=m;

stde= std(s);
feature_epoch(o,7)=stde;

rmdd = rms(s);
feature_epoch(o,8)=rmdd;

meanab= mad(s);
feature_epoch(o,9)=meanab;

inte = iqr(s);
feature_epoch(o,10)=inte;

% entr = entropy(s);
% feature_epoch(o,11)=entr;

Nx = length(s);
nsc = floor(Nx/4.5);
nov = floor(nsc/2);
nff = max(256,2^nextpow2(nsc));

t = pwelch(s,hamming(nsc),nov,nff);
feature_epoch(o,11)=sum(t);

Mobility=sqrt(var(diff(s))/var(s));
Complexity=sqrt(var(diff(diff(s)))/var(s))/sqrt(var(diff(s))/var(s));

feature_epoch(o,12)=Mobility;
feature_epoch(o,13)=Complexity;

pbandDelta = bandpower(s,100,[0.4 4]);
pbandTheta = bandpower(s,100,[4 8]);
pbandAlfa = bandpower(s,100,[8 13]);

feature_epoch(o,14)=pbandDelta;
feature_epoch(o,15)=pbandTheta;
feature_epoch(o,16)=pbandAlfa;

feature_epoch(o,17)=max(s)-min(s);

%MEAN FEATURE


for w=1:17

feature(1,w)=sum(feature_epoch(:,w))/length(feature_epoch(:,1));

end

end
switch i
    case 1
       featuren=feature;
save('featuren1.mat','featuren');
    case 2
       featuren=feature;
save('featuren2.mat','featuren');
        
    case 3
       featuren=feature;
save('featuren3.mat','featuren');
    case 4
       featuren=feature;
save('featuren4.mat','featuren');
    case 5
       featuren=feature;
save('featuren5.mat','featuren');
    case 6
       featuren=feature;
save('featuren6.mat','featuren');
    case 7
       featuren=feature;
save('featuren7.mat','featuren');
    case 8
       featuren=feature;
save('featuren8.mat','featuren');
    case 9
       featuren=feature;
save('featuren9.mat','featuren');
    case 10
       featuren=feature;
save('featuren10.mat','featuren');
    case 11
       featuren=feature;
save('featuren11.mat','featuren');
    case 12
       featuren=feature;
save('featuren12.mat','featuren');
    case 13
       featuren=feature;
save('featuren13.mat','featuren');
    case 14
       featuren=feature;
save('featuren14.mat','featuren');
    case 15
       featuren=feature;
save('featuren15.mat','featuren');
    case 16
       featuren=feature;
save('featuren16.mat','featuren');
    otherwise
        disp('hata')
end

 feature=[];
 feature_epoch=[];
 j=0;
 k=0;
end
clc
%Merge All data
tum_hasta = [f1;f2;f3;f4;f5;f6;f7;f8;f9;f10;f11;f12;f13;f14;f15;f16;f17;f18;f19;f20;f21;f22;f1;fn2;fn3;fn4;fn5;fn6;fn7;fn8;fn9;fn10;fn11;fn12;fn13;fn14;fn15;fn16];

%If diase occur resul twill be 1 otherwise  0
for i=1:1:22
tum_hasta(i,18)=1;
end
for i=23:1:38
tum_hasta(i,18)=0;
end
tum_hasta= normalize(tum_hasta);
save('sag_has_all_featuree.mat','tum_hasta');
%% TEST TRAIN 

L=length(sag_has_all_feature(1,:));
n = length(sag_has_all_feature(:,L));
partitionPart = cvpartition(n,'Holdout',0.3); % Nonstratified partition
index_of_train = training(partitionPart); 
table_of_train = sag_has_all_feature(index_of_train,:); % train kısmı toplam veri 
index_of_test = test(partitionPart);
table_of_test = sag_has_all_feature(index_of_test,:); 

EgitimHyp= table_of_train(:,L);
EgitimFeature= table_of_train();
EgitimFeature(:,L)= [];
TestHyp= table_of_test(:,L);
TestFeature= table_of_test;
TestFeature(:,L)= [];
%% SVM
template_svm = templateSVM(...
 'KernelFunction', 'polynomial', ...
 'PolynomialOrder', 2, ...
 'KernelScale', 'auto', ...
 'BoxConstraint', 1, ...
 'Standardize', true);
classificationSVM = fitcecoc( EgitimFeature, EgitimHyp, ...
 'Learners', template_svm, ...
 'Coding', 'onevsone', ...
 'ClassNames', [0 ; 1]);

Testsonuc_SVM=predict(classificationSVM,TestFeature);
Accuracy_SVM = sum((Testsonuc_SVM == TestHyp))/length(TestHyp)*100;
confusion_matrix_SVM = confusionmat(TestHyp,Testsonuc_SVM);
sensivity_SVM=confusion_matrix_SVM(1,1)/sum(confusion_matrix_SVM(:,1))*100;
specificity_SVM=confusion_matrix_SVM(2,2)/sum(confusion_matrix_SVM(:,2))*100
confusionchart(TestHyp,Testsonuc_SVM)
%% KNN
 classificationKNN = fitcknn(...
 EgitimFeature, ...
 EgitimHyp, ...
 'Distance', 'Euclidean', ...
 'Exponent', [], ...
 'NumNeighbors', 2, ...
 'DistanceWeight', 'SquaredInverse', ...
 'Standardize', true, ...
 'ClassNames', [0; 1]); 
Testsonuc_KNN =predict(classificationKNN,TestFeature);
Accuracy_KNN = sum((Testsonuc_KNN == TestHyp))/length(TestHyp)*100;
confusion_matrix_KNN = confusionmat(TestHyp,Testsonuc_KNN);
sensivity_KNN=confusion_matrix_KNN(1,1)/sum(confusion_matrix_KNN(:,1))*100;
specificity_KNN=confusion_matrix_KNN(2,2)/sum(confusion_matrix_KNN(:,2))*100;
confusionchart(TestHyp,Testsonuc_KNN)

%% DT
 classificationTree = fitctree(...
 EgitimFeature, ...
 EgitimHyp, ...
 'SplitCriterion', 'gdi', ...
 'MaxNumSplits', 20, ...
 'Surrogate', 'off', ...
 'ClassNames', [0; 1]);
Testsonuc_TREE=predict(classificationTree,TestFeature);
Accuracy_TREE = sum((Testsonuc_TREE == TestHyp))/length(TestHyp)*100;
confusion_matrix_TREE = confusionmat(TestHyp,Testsonuc_TREE);
sensivity_TREE=confusion_matrix_TREE(1,1)/sum(confusion_matrix_TREE(:,1))*100;
specificity_TREE=confusion_matrix_TREE(2,2)/sum(confusion_matrix_TREE(:,2))*100;
confusionchart(TestHyp,Testsonuc_TREE)
%% All Result


SONUC =[Accuracy_SVM sensivity_SVM specificity_SVM ;
 Accuracy_KNN sensivity_KNN specificity_KNN ;
 Accuracy_TREE sensivity_TREE specificity_TREE]; 
save('SONUC.mat','SONUC');
