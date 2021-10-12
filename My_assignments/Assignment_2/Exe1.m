clc;clear;
workspace;
disp("Starting script. Complete script running time is about 47 minutes! So be patient...");
% Load data - Split data
data=load('airfoil_self_noise.dat');
preprocess=1;
[trainData,valData,testData]=split_data(data,preprocess);


% Evaluation function, we are using Rsquared
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);


% generate fis with grid partition
fis(1) = genfis1(trainData,2,'gbellmf','constant');
fis(2) = genfis1(trainData,3,'gbellmf','constant');
fis(3) = genfis1(trainData,2,'gbellmf','linear');
fis(4) = genfis1(trainData,3,'gbellmf','linear');

models_array=NaN(4,4);
for i = 1:4
    % ANFIS training
    [trnFis,trnError,~,valFis,valError] = anfis(trainData,fis(i),[100 0 0.01 0.9 1.1],[],valData);
    % 100 epochs of training
    % Membership functions plots
    message = "TSK model ";
    %convert index i of model to string , so that the strings can be
    %concatanated
    numberTSK = int2str(i);
    message = strcat(message,numberTSK);
    message = strcat(message," membership functions for input ");
    %plot membership functions in numbered figures using evalfis
    for j = 1:size(trainData,2)-1
        input = int2str(j);
        Title = strcat(message,input);
        figure(i*100 + j);
        plotmf(valFis,'input',j);
        title(Title);
    end

    % Learning Curve plot, numbered figures as well
    figure(i*100 + 10 + i);
    plot([trnError valError],'LineWidth',2); grid on;
    xlabel('# of Iterations'); ylabel('Error');
    legend('Training Error','Validation Error');
    message = "TSK model ";
    message = strcat(message,numberTSK);
    message = strcat(message," learning curve");
    title(message);
    Y = evalfis(testData(:,1:end-1),valFis);
    %evaluate R squared using our function handle
    Rsquared = Rsq(Y,testData(:,end));
    %calculate the rest of the perfomance indicators
    RMSE = sqrt(mse(Y,testData(:,end)));
    NMSE = 1 - Rsquared;
    NDEI = sqrt(NMSE);
    models_array(i,:) = [Rsquared; RMSE; NMSE; NDEI];
    
    % Prediction Error plot
    predictionError = testData(:,end) - Y;
    figure(100*i + 20);
    plot(predictionError,'LineWidth',2); grid on;
    message = "TSK model ";
    message = strcat(message,numberTSK);
    message = strcat(message," prediction error");
    xlabel('input');ylabel('Error');
    title(message);
end

% Results Table
variable_names={'TSK_model_1','TSK_model_2','TSK_model_3','TSK_model_4'};
row_names={'Rsquared','RMSE','NMSE','NDEI'};
models_array = models_array';
models_array = array2table(models_array,'VariableNames',variable_names,'RowNames',row_names)
disp("End of script");
