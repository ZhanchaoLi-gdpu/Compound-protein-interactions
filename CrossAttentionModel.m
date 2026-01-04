% clear
% clc

load('...\DrugTargetInformation.mat')
clear ans DrugName DrugSmiles HyperGraph_ProteinEdge
clear HyperGraph_DrugEdge TargetSeq

load muprotein.mat
load mudrug.mat

% construct positive example Feature
PosFeature = zeros(height(DrugTargetInteract),size(mudrug,2)*2);
PosSampAll = cell(height(DrugTargetInteract),1);
for i = 1 : height(DrugTargetInteract)
    
    lineD = find(strcmp(DrugID,DrugTargetInteract{i,1}));
    lineP = find(strcmp(TargetID,DrugTargetInteract{i,4}));
    if  ~isempty(lineD) && ~isempty(lineP)
        PosFeature(i,:) = [mudrug(lineD,:) muprotein(lineP,:)];
    else

    end

    PosSampAll{i,1} = [DrugTargetInteract{i,1} '-' DrugTargetInteract{i,4}];
end
clear i lineD lineP

% generate negative samples with size equal to positive samples
% by randomly matching drugs and proeins, and ensure that these
% negative samples are not included in the positive samples.
LeCoMu = lcm(length(DrugID),length(TargetID));
drug = repmat(DrugID,LeCoMu/length(DrugID),1);
prot = repmat(TargetID,LeCoMu/length(TargetID),1);
SampAll = cell(LeCoMu,1);
for i = 1 : length(drug)
    SampAll{i,1} = [drug{i,1} '-' prot{i,1}];
end
[NegSampAll,ia] = setdiff(SampAll,PosSampAll);
[Test,Train] = crossvalind('LeaveMOut',length(NegSampAll),size(PosSampAll,1));
NegSampSel = NegSampAll(Train,:);
clear LeCoMu drug prot i ia 

% Negative sample feature with same size of positive sample feature 
NegFeature = zeros(height(NegSampSel),size(mudrug,2)*2);
for i = 1 : length(NegSampSel)
    temp = regexp(NegSampSel{i,1},'-','split');

    lineD = find(strcmp(DrugID,temp{1,1}));
    lineP = find(strcmp(TargetID,temp{1,2}));
    if  ~isempty(lineD) && ~isempty(lineP)
        NegFeature(i,:) = [mudrug(lineD,:) muprotein(lineP,:)];
    else

    end

end
Y = [ones(height(PosSampAll),1);2*ones(length(NegSampSel),1)];
Y = categorical(Y);

% test unknown sample
UnkSampTest = NegSampAll(Test,:);
UnknownFeature = zeros(height(UnkSampTest),size(mudrug,2)*2);
for i = 1 : length(UnkSampTest)
    temp = regexp(UnkSampTest{i,1},'-','split');

    lineD = find(strcmp(DrugID,temp{1,1}));
    lineP = find(strcmp(TargetID,temp{1,2}));
    if  ~isempty(lineD) && ~isempty(lineP)
        UnknownFeature(i,:) = [mudrug(lineD,:) muprotein(lineP,:)];
    else

    end

end
YTest = 2*ones(length(UnkSampTest),1);
YTest = categorical(YTest);
clear i temp lineD lineP SampAll NegSampAll 


classNames = categories(Y);
numSamples = length(Y);
numClasses = 2;   % 分类类别数
numHeads = 8;

Feature = [PosFeature;NegFeature];
numFeatures1 = size(Feature,2)/2; % X1特征维度
numFeatures2 = size(Feature,2)/2; % X2特征维度
%clear PosFeature NegFeature

% 网络结构定义
input1 = featureInputLayer(numFeatures1,'Name','inpu1');
input2 = featureInputLayer(numFeatures2,'Name','inpu2');

% 修正注意力层连接
attention1 = attentionLayer(numHeads,'Name','at1');
attention2 = attentionLayer(numHeads,'Name','at2');

net = dlnetwork;

net = addLayers(net,input1);
net = addLayers(net,input2);
net = addLayers(net,attention1);
net = addLayers(net,attention2);

% 确保所有输入端口连接
net = connectLayers(net,'inpu1','at1/query');
net = connectLayers(net,'inpu2','at1/key');
net = connectLayers(net,'inpu2','at1/value');
net = connectLayers(net,'inpu2','at2/query');
net = connectLayers(net,'inpu1','at2/key');
net = connectLayers(net,'inpu1','at2/value');

% 合并与分类头
concat = concatenationLayer(1,2,'Name','concat');
net = addLayers(net,concat);
net = connectLayers(net,'at1','concat/in1');
net = connectLayers(net,'at2','concat/in2');

classifier = [
    fullyConnectedLayer(256,'Name','fc1')
    reluLayer('Name','relu1')
    dropoutLayer(0.25)
    fullyConnectedLayer(128,'Name','fc2')
    reluLayer('Name','relu2')
    dropoutLayer(0.25)
    fullyConnectedLayer(numClasses,'Name','fc3')
    softmaxLayer('Name','softmax')
    ];

net = addLayers(net,classifier);
net = connectLayers(net,'concat','fc1');

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 512, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'none');  %'training-progress'

% 模型训练
dsX1Train = arrayDatastore(Feature(:,1:numFeatures1));
dsX2Train = arrayDatastore(Feature(:,1+numFeatures1:end));
dsTTrain = arrayDatastore(Y(:,1));
dsTrain = combine(dsX1Train,dsX2Train,dsTTrain);
netTrained = trainnet(dsTrain,net,"crossentropy",options);

Votes = minibatchpredict(netTrained,dsTrain);
Y_hat = scores2label(Votes,classNames);

ConMat = confusionmat(Y,Y_hat,'order',categorical([1,2]));
TP = ConMat(1,1);
TN = ConMat(2,2);
FN = ConMat(1,2);
FP = ConMat(2,1);
Acc = (TP + TN)/(TP+FN+TN+FP);
Sen = TP/(TP+FN);
Spe = TN/(TN+FP);
Pre = TP/(TP + FP);
Mcc = (TP*TN-FP*FN)/sqrt((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP));
[TXR,TYR,TTR,AUCR] = perfcurve(Y,Votes(:,1),1);
[TXP,TYP,TTP,AUCP] = perfcurve(Y,Votes(:,1),1,'xcrit','reca','ycrit','prec');

% prediction unknown samples
dsX1Test = arrayDatastore(UnknownFeature(:,1:numFeatures1));
dsX2Test = arrayDatastore(UnknownFeature(:,1+numFeatures1:end));
dsTTest = arrayDatastore(YTest(:,1));
dsTest = combine(dsX1Test,dsX2Test,dsTTest);
VotesTest = minibatchpredict(netTrained,dsTest);
YTestPred = scores2label(VotesTest,classNames);

label = find(1 == double(YTestPred));
IdenComProtInter = UnkSampTest(label);


save LarSccPredResults Test Train Acc Sen Spe Pre Mcc TXR TYR TXP TYP AUCR ...
     AUCP VotesTest YTestPred label IdenComProtInter





