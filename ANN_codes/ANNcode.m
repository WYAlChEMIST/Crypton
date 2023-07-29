%%
%This code is developed by K. Jitapunkul and ALCHEMIST team
%train data with two hidden layer ANN
clear x y numfolds c h net tr Hmin Hmax
clear Testperf_record Trainperf_record train_idx test_idx;
clear xTrain yTrain xTest yTest ypredict ytrain_net ycom ycomtr
clear Testperf Trainperf testRcal testR testR2 trainRcal trainR trainR2
clear stopcrit bestepoch runtime

load ann_input_output.mat
%%
rng('default');
x = ann_input; %Please specify the ann_input as scan rate, concentration, and potential windows (row basis)
y = ann_output_neg; %Please specify the ann_output_[yourchoice] as concatenate potential and current series (you can select [yourchoice] as full, neg, or pos)
numfolds = 5; %number of folds
c=cvpartition(size(y,2),'KFold',numfolds); %partition data for k-fold cross validation (test 20% by default)

%Table to store results
Testperf_record=zeros(numfolds,1);
Trainperf_record=zeros(numfolds,1);

Hmin= 1; %number of minimum hidden nodes
Hmax= 20; %number of maximum hidden nodes

for h = Hmin:Hmax
    net = fitnet([h,200],'trainrp');
   
for i=1:numfolds
    %divide data for train and test data for each fold
    fieldfold=sprintf('Fold%d',i);
    fieldfold1=sprintf('Node%d',h);
    train_idx.(fieldfold1)(:,i)=c.training(i);
    test_idx.(fieldfold1)(:,i)=c.test(i);
    xTrain.(fieldfold1).(fieldfold) = x(:,train_idx.(fieldfold1)(:,i));
    yTrain.(fieldfold1).(fieldfold) = y(:,train_idx.(fieldfold1)(:,i));
    xTest.(fieldfold1).(fieldfold) = x(:,test_idx.(fieldfold1)(:,i));
    yTest.(fieldfold1).(fieldfold) = y(:,test_idx.(fieldfold1)(:,i));
     
   %Set Division function
   net.divideFcn='dividerand'; %random division
   % Setup Division of Data for Training, Validation, Testing
   net.divideParam.trainRatio = 80/100;
   net.divideParam.valRatio = 10/100;
   net.divideParam.testRatio = 10/100;
   
   %Default learning rate  = 0.01
   
   %set function of hidden layers
   net.layers{1}.transferFcn = 'tansig' ;
   net.layers{2}.transferFcn = 'radbasn' ;
   net.layers{3}.transferFcn = 'purelin';
   
   %train the model
   [netre, tr]    = train(net, xTrain.(fieldfold1).(fieldfold), yTrain.(fieldfold1).(fieldfold));
   ypredict = netre(xTest.(fieldfold1).(fieldfold));
   ytrain_net = netre(xTrain.(fieldfold1).(fieldfold));
   stopcrit{i,h}  = tr.stop;
   bestepoch(i,h) = tr.best_epoch;
   runtime(i,h) = tr.time(end);
   Testperf = perform(net,yTest.(fieldfold1).(fieldfold),ypredict);
   Testperf_record(i,h) = Testperf; %record value of MSE (Loss calculation of test)
   Trainperf = perform(net,yTrain.(fieldfold1).(fieldfold),ytrain_net);
   Trainperf_record(i,h) = Trainperf; %record value of MSE (Loss calculation of train)
   testRcal=corrcoef(ypredict,yTest.(fieldfold1).(fieldfold));
   testR(i,h)=testRcal(1,2);
   testR2(i,h)=(testR(i,h))^2;
   trainRcal=corrcoef(ytrain_net,yTrain.(fieldfold1).(fieldfold));
   trainR(i,h)=trainRcal(1,2);
   trainR2(i,h)=(trainR(i,h))^2;
   testRMSE(i,h)=sqrt(mse(ypredict,yTest.(fieldfold1).(fieldfold)));
   trainRMSE(i,h)=sqrt(mse(ytrain_net,yTrain.(fieldfold1).(fieldfold)));
   
   netcom.(fieldfold1).(fieldfold) = netre;
   ycom.(fieldfold1).(fieldfold) = ypredict;
   ycomtr.(fieldfold1).(fieldfold) = ytrain_net;
   trcom.(fieldfold1).(fieldfold) = tr;
   
    net = init(net);
end
end

disp('ANN training completed!')