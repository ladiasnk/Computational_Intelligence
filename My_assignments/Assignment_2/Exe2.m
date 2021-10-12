clc;clear;
disp("Script begun...");

% Read data and normalise
data = csvread('train.csv',1,0);
data = normaliseData(data);
%Rsquared function handle
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% Define grid search parameters
a = 5;          
b = 7;
% First param is num of features, second param is clusters radius
gridSearchParameters = zeros(a,b,2);
%Test values for features : 5,10,15,20,25
%Test values for cluster radius: 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8
gridSearchParameters(1,:,1) = 5;
gridSearchParameters(2,:,1) = 10;
gridSearchParameters(3,:,1) = 15;
gridSearchParameters(4,:,1) = 20;
gridSearchParameters(5,:,1) = 25;
gridSearchParameters(:,1,2) = 0.2;
gridSearchParameters(:,2,2) = 0.3;
gridSearchParameters(:,3,2) = 0.4;
gridSearchParameters(:,4,2) = 0.5;
gridSearchParameters(:,5,2) = 0.6;
gridSearchParameters(:,6,2) = 0.7;
gridSearchParameters(:,7,2) = 0.8;
% errors array to save all the Rsquared and RMSE errors using a third
% dimension
errors = zeros(a,b,2);

% Define k-fold cross validation parameter to be used 
k = 5;
% tic toc the process to know what's the time duration of the training 
tic
% Grid search
for i = 1:a
    for j = 1:b
        numOfFeatures = gridSearchParameters(i,j,1);
        ra = gridSearchParameters(i,j,2);
        
        crossValidationErrors = zeros(k,2);
        
        % k-fold cross validation
        for iteration = 1:k
            [trainData,checkData,testData] = crossValidationDatasets(k,data,iteration);
            
            % Feature selection
            [idx,weights] = relieff( trainData(:,1:end-1), trainData(:,end),5);
            
            %idx = idx( 1:numOfFeatures );
            trainDataFS = trainData( :, idx(1:numOfFeatures) );
            trainDataFS = [ trainDataFS trainData( :, end)];
            
            checkDataFS = checkData( :, idx(1:numOfFeatures) );
            checkDataFS = [ checkDataFS checkData( :, end) ];
            
            testDataFS = testData( :, idx(1:numOfFeatures) );
            testDataFS = [ testDataFS testData( :, end) ];
            
            
            % genfis2 (SC)
            My_fis = genfis2(trainDataFS(:,1:end-1),trainDataFS(:,end),ra);
            
            
            % Training
            disp("Training begun...");
            [trainFis,trainError,~,valFis,valError] = anfis(trainDataFS,My_fis,[100 0 0.01 0.9 1.1],[],checkDataFS);
            figure(iteration); % each iteration produces its own figure
            plot([trainError valError],'LineWidth',2); grid on;
            xlabel('# of Iterations'); ylabel('Error');
            legend('Training Error','Validation Error');
            title('ANFIS Hybrid Training - Validation');
            disp("Training ending");
            %now compute all the necessary metrics
            Y = evalfis(testDataFS(:,1:end-1),valFis);
            R2 = Rsq(Y,testDataFS(:,end));
            RMSE = sqrt(mse(Y,testDataFS(:,end)));
            NMSE = 1 - R2;
            NDEI = sqrt(NMSE);
            disp("End of Training");
          
            % Save errors for cross validation
            crossValidationErrors(iteration,1) = R2;
            crossValidationErrors(iteration,2) = RMSE;
        end
        % Find average of cross validation errors and save it
        tempErrorR2 = sum( crossValidationErrors(:,1) ) / k;
        tempErrorRMSE = sum( crossValidationErrors(:,2) ) / k;
        
        errors(i,j,1) = tempErrorR2;
        errors(i,j,2) = tempErrorRMSE;
        
        
    end
end
toc
time=tic-toc;
errors
disp("End of script");