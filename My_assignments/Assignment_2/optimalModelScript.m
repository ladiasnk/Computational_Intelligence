clc;clear;
disp("Script begun...");

% Read data and normalise
data = csvread('train.csv',1,0);

data = normaliseData(data);
%Rsquared function handle
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% Optimal Model
numOfFeatures = 25;
radius = 0.7;

% k-fold cross validaiton
k = 5;

% we crossvalidate R2 and RMSE
crossValErrors = zeros(k,2);
        
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
    fis = genfis2(trainDataFS(:,1:end-1),trainDataFS(:,end),radius);
    
    % Plot mf before training
    titleBefore = "Optimal TSK model membership functions for input ";
    figure(1);
    plotmf(fis,'input',1);
    title1 = strcat(titleBefore,'1');
    title1 = strcat(title1,' before training');
    title(title1);
    
    figure(2);
    plotmf(fis,'input',size(trainDataFS,2)-1);
    num = int2str(size(trainDataFS,2)-1);
    title2 = strcat(titleBefore,num);
    title2 = strcat(title2,' before training');
    title(title2);
    
    % Training
    disp("Start of Training");
    [trnFis,trnError,~,valFis,valError] = anfis(trainDataFS,fis,[250 0 0.01 0.9 1.1],[],checkDataFS);
    disp("End of Training");
    
    % Learning Curve plot
    figure(3);
    plot([trnError valError],'LineWidth',2); grid on;
    xlabel('# of Iterations'); ylabel('Error');
    legend('Training Error','Validation Error');
    title('Optimal TSK model learning curve');
    
    % Plot mf after training
    titleAfter = "Optimal TSK model membership functions for input ";
    figure(4);
    plotmf(valFis,'input',1);
    title1 = strcat(titleAfter,'1');
    title1 = strcat(title1,' after training');
    title(title1);
    
    figure(5);
    plotmf(valFis,'input',size(trainDataFS,2)-1);
    num = int2str(size(trainDataFS,2)-1);
    title2 = strcat(titleAfter,num);
    title2 = strcat(title2,' after training');
    title(title2);
    
    Y = evalfis(testDataFS(:,1:end-1),valFis);
    R2 = Rsq(Y,testDataFS(:,end));
    RMSE = sqrt(mse(Y,testDataFS(:,end)));
    
    % Predictions plot
    figure(7)
    plot(Y,'O'); grid on;
    xlabel('input');
    legend('Prediction');
    title('Model predictions');
    
    % Ground truth
    figure(8)
    plot(testDataFS(:,end),'O'); grid on;
    xlabel('input');
    legend('Ground truth');
    title('Ground truth');
    
    % Prediction Error plot
    predictionError = testDataFS(:,end) - Y;
    figure(9);
    plot(predictionError,'O'); grid on;
    xlabel('input');ylabel('Error');
    legend('Prediction Error');
    title('Optimal TSK model prediction error');
    
    % Save cross validation errors
    crossValErrors(iteration,1) = R2;
    crossValErrors(iteration,2) = RMSE;
end
% Find average of cross validation errors and save it
R2 = sum( crossValErrors(:,1) ) / k
RMSE = sum( crossValErrors(:,2) ) / k
NMSE = 1 - R2;
NDEI = sqrt(NMSE);

errors = [R2; RMSE; NMSE; NDEI]

disp("End of script");