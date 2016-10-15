%Opening and processing a real database
file = fopen('C:\Work\COURSES\SEM-1\MACHINE LEARNIN\ML Project-2\MQ2007\Querylevelnorm.txt');
formating = ['%f %*s' repmat('%*d:%f',1,46) '%*[^\n]']; 
filescanning = (textscan(file,formating)); 
fclose(file);

A = cell2mat(filescanning);

%defining the TargetSet and the Training/Validation/Testing Set
Main = A(:,2:47);   %Training/Validation/Testing Set
Target = A(:,1);             %Target Set
Row_Seg= size(Target);

%Setting the size of the Training/Validation/Testing Sets.
Train_size= round(.8 * Row_Seg);
Validate_size = round(.1 * Row_Seg);
Test_size = round(.1 * Row_Seg);

%Segregating the Training/Validation/Testing Sets.
Train_set = Main(1:Train_size,:);
Validate_set = Main(Train_size+1:Train_size+Validate_size,:);
Test_set = Main(Train_size+Validate_size+1:end,:);

%Segregating the Target Sets.
T_Train = Target(1:Train_size,:);
T_Validate = Target(Train_size+1:Train_size+Validate_size,:);
T_Test = Target(Train_size+Validate_size+1:end,:);

%Creating the design matrix for the Training/Validation/Testing Set
M1=10;
Sigma1_T = eye(46);

%Assigning the first element of the design matrix-Training as ones
One_designMatTrain = ones(length(Train_set),1);
designMatTrain = [One_designMatTrain];

%Assigning the first element of the design matrix-Validation as ones
One_designMatValidate = ones(length(Validate_set),1);
designMatValidate = [One_designMatValidate];

%Assigning the first element of the design matrix-Testing as ones
One_designMatTest = ones(length(Test_set),1);
designMatTest = [One_designMatTest];

%Setting the KMean clusters for the Training Sets
[~,mu1_Train] = kmeans(Train_set,M1-1);


%Assigning the first element of the design matrix-Testing as ones
One_muTrain = ones(1,length(mu1_Train));



%For real data
for i = 1:M1-1 
    %design matrix for Training Set
    for j= 1:length(Train_set)
        xT = Train_set(j,:)- mu1_Train(i,:);
        phi_Train(j,:) = exp(-1/2 * xT* Sigma1_T * (xT'));
        dMat = phi_Train;
    end 
    designMatTrain=[designMatTrain dMat];
    
    %Finding the weight for the Training Set
    Lambda = 0.00000073;
    designMatrows_Train = size(designMatTrain'*designMatTrain,1);
    designMatcols_Train = size(designMatTrain'*designMatTrain,2);
    ID_Training = eye(designMatrows_Train,designMatcols_Train);
    ID_Train = Lambda*ID_Training;
    w1 = inv(ID_Train+(designMatTrain'*designMatTrain))*(designMatTrain'*T_Train);      
        
    %design matrix for Validate Set
    for l= 1:length(Validate_set)
        yT = Validate_set(l,:)- mu1_Train(i,:);
        phi_Validate(l,:) = exp(-1/2 * yT* Sigma1_T * (yT'));
        dMat_V = phi_Validate;
    end 
    designMatValidate=[designMatValidate dMat_V];
    
end

%Training the 
 %Finding the rmsError for Training Set
    Ed = 0.5 * ((designMatTrain*w1-T_Train)' * (designMatTrain*w1-T_Train));
    Ew = 0.5 * (w1'*w1);
    Error = Ed + (Lambda*Ew);
    Nv = size(Train_set,1);
    Erms = sqrt(2* (Error/Nv));
    
  %Finding the rmsError for Validating Set  
    Ed_V = 0.5 * ((designMatValidate*w1-T_Validate)' * (designMatValidate*w1-T_Validate));
    Ew_V = 0.5 * (w1'*w1);
    Error_V = Ed_V;
    Nv_V = size(Validate_set,1);
    Erms_V = sqrt(2* (Error_V/Nv_V));

for k = 1 : M1
    Sigma1(:,:,k) =  Sigma1_T;
end


% To process the synthetic data
X = x'; % Order the data for processing
Syn_length = length(X);

%length of the Training/Validation/Testing sets of Synthetic data
Syn_Train_size = round(.8 * Syn_length);
Syn_Validate_size = round(.1 * Syn_length);
Syn_Test_size = round(.1 * Syn_length);

%Seperated sets for Training/Validation and Testing for Synthetic data X
Syn_Train_set = X(1:Syn_Train_size,:);
Syn_Validate_set = X(Syn_Train_size+1:Syn_Train_size+Syn_Validate_size,:);
Syn_Test_set = X(Syn_Train_size+Syn_Validate_size+1:end,:);

%Seperated sets for Training/Validation and Testing for Synthetic Target t
Syn_T_Train = t(1:Syn_Train_size,:);
Syn_T_Validate = t(Syn_Train_size+1:Syn_Train_size+Syn_Validate_size,:);
Syn_T_Test = t(Syn_Train_size+Syn_Validate_size+1:end,:);

%Creating the design matrix for the Training/Validation/Testing Set
M2=10;
Sigma2_T = eye(10);

%Assigning the first element of the design matrix-Training as ones
Syn_One_designMatTrain = ones(length(Syn_Train_set),1);
Syn_designMatTrain = [Syn_One_designMatTrain];

%Assigning the first element of the design matrix-Validation as ones
Syn_One_designMatValidate = ones(length(Syn_Validate_set),1);
Syn_designMatValidate = [Syn_One_designMatValidate];

%Assigning the first element of the design matrix-Testing as ones
Syn_One_designMatTest = ones(length(Syn_Test_set),1);
Syn_designMatTest = [Syn_One_designMatTest];

%Setting the KMean clusters for the Training Sets
[~,Syn_mu2_Train] = kmeans(Syn_Train_set,M2-1);


%Assigning the first element of the design matrix-Testing as ones
Syn_One_muTrain = ones(1,length(Syn_mu2_Train));


%For synthetic data
for m = 1:M2-1 
    %design matrix for Training Set
    for n= 1:length(Syn_Train_set)
        Syn_xT = Syn_Train_set(n,:)- Syn_mu2_Train(m,:);
        Syn_phi_Train(n,:) = exp(-1/2 * Syn_xT* Sigma2_T * (Syn_xT'));
        Syn_dMat = Syn_phi_Train;
    end 
    Syn_designMatTrain=[Syn_designMatTrain Syn_dMat];
    
    %Finding the weight for the Training Set
    Lambda2 = 0.00000073;
    Syn_designMatrows_Train = size(Syn_designMatTrain'*Syn_designMatTrain,1);
    Syn_designMatcols_Train = size(Syn_designMatTrain'*Syn_designMatTrain,2);
    Syn_ID_Training = eye(Syn_designMatrows_Train,Syn_designMatcols_Train);
    Syn_ID_Train = Lambda2*Syn_ID_Training;
    w2 = inv(Syn_ID_Train+(Syn_designMatTrain'*Syn_designMatTrain))*(Syn_designMatTrain'*Syn_T_Train);
        
    %design matrix for Validate Set
    for o= 1:length(Syn_Validate_set)
        Syn_yT = Syn_Validate_set(o,:)- Syn_mu2_Train(m,:);
        Syn_phi_Validate(o,:) = exp(-1/2 * Syn_yT* Sigma2_T * (Syn_yT'));
        Syn_dMat_V = Syn_phi_Validate;
    end 
    Syn_designMatValidate=[Syn_designMatValidate Syn_dMat_V];
    
end

%Training the 
 %Finding the rmsError for Training Set
    Syn_Ed = 0.5 * ((Syn_designMatTrain*w2 - Syn_T_Train )' * (Syn_designMatTrain*w2-Syn_T_Train ));
    Syn_Ew = 0.5 * (w2'*w2);
    Syn_Error = Syn_Ed ;
    Syn_Nv = size(Syn_Train_set,1);
    Syn_Erms = sqrt(2* (Syn_Error/Syn_Nv));
    
  %Finding the rmsError for Validating Set  
    Syn_Ed_V = 0.5 * (( Syn_designMatValidate*w2-Syn_T_Validate)' * (Syn_designMatValidate*w2 - Syn_T_Validate));
    Syn_Ew_V = 0.5 * (w2'*w2);
    Syn_Error_V = Syn_Ed_V;
    Syn_Nv_V = size(Syn_Validate_set,1);
    Syn_Erms_V = sqrt(2* (Syn_Error_V/Syn_Nv_V));

for p = 1 : M2
    Sigma2(:,:,p) =  Sigma2_T;
end





%data to be saved in the proj2.mat file
trainPer1 = Erms;
trainPer2 = Syn_Erms;
validPer1 = Erms_V;
validPer2 = Syn_Erms_V;
mu1 = [One_muTrain; mu1_Train]';
mu2 = [Syn_One_muTrain; Syn_mu2_Train]';
trainInd1 = (1:length(Train_set))';
trainInd2 = (1:length(Syn_Train_set))';
validInd1 = ((length(Train_set))+1:(length(Validate_set)+length(Train_set)))';
validInd2 = ((length(Syn_Train_set))+1:(length(Syn_Validate_set)+length(Syn_Train_set)))';
lambda1 =Lambda;
lambda2 =Lambda2;


% Training for Stochastic Gradient - real dataset
 
eta1_gd= 0.01;
w01 = zeros(M1,1);
dw1_ = zeros(M1,1);
dw1_mid = zeros(M1,1);


i1_gd = 1;
iteration1 = 5;
while i1_gd < iteration1
for j1 = 1: length(T_Train)
    Ed1_gd= -1*(T_Train(j1,:)- dw1_mid'*designMatTrain(j1,:)')*(designMatTrain(j1,:));
    Ew1_gd= dw1_mid;
    Error1_gd= Ed1_gd + (Lambda*Ew1_gd');
    del_w1_syn= -1*eta1_gd*Error1_gd;
    dw1_mid = dw1_mid + del_w1_syn';
    dw1_= [dw1_ del_w1_syn'];
    
end
 dw1=dw1_(1:end,2:end);
 i1_gd = i1_gd +1;
end

for j1_gd = 1:((i1_gd-1) * length(T_Train))
    eta1(j1_gd) = eta1_gd;
end


% Training for Stochastic Gradient - Synthetic dataset
 
eta2_gd= 0.1;
w02 = zeros(M2,1);
dw2_ = zeros(M2,1);
dw2_mid = zeros(M2,1);


i2_gd = 1;
iteration2 = 15;
while i2_gd < iteration2
for i2 = 1: length(Syn_T_Train)
    Ed_gd= -1*(Syn_T_Train(i2,:)- dw2_mid'*Syn_designMatTrain(i2,:)')*(Syn_designMatTrain(i2,:));
    Ew_gd= dw2_mid;
    Error_gd= Ed_gd + (Lambda2*Ew_gd');
    del_w2_syn= -1*eta2_gd*Error_gd;
    dw2_mid = dw2_mid + del_w2_syn';
    dw2_= [dw2_ del_w2_syn'];
    
end
 dw2=dw2_(1:end,2:end);
 i2_gd = i2_gd + 1;
end

for j2_gd = 1:((i2_gd-1) * length(Syn_T_Train))
    eta2(j2_gd) = eta2_gd;
end


UbitName = 'rnayak';
personNumber = 50169647;

save('proj2.mat', 'UbitName', 'personNumber', 'M1','M2','mu1','mu2','Sigma1','Sigma2', 'lambda1','lambda2', 'trainInd1','trainInd2', 'w1','w2','validInd1','validInd2','trainPer1','trainPer2','validPer1','validPer2','eta2','dw2','w02','eta1','dw1','w01');
