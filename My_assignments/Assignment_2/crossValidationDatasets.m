function [trainData,checkData,testData] = crossValidationDatasets(k,data,iteration)
%   crossValidationDatasets
%   this function returns the datasets required for k-fold cross validation
%   data is split 60%/20%/20% into trainData,checkData,testData respectivly

    n = size(data,1);

    % find testData and remove them from dataset
    if( iteration ~= k )
        % if it is not the last iteration
        testData = data( round(n/k)*(iteration-1) + 1 : round(n/k)*iteration , :);
        trainCheckDatasets = data;
        trainCheckDatasets( round(n/k)*(iteration-1) + 1 : round(n/k)*iteration , : ) = [];
    else
        % if it is the last iteration
        % prevents exceding the length of the array due to rounding
        testData = data( round(n/k)*(iteration-1) + 1 : end , :);
        trainCheckDatasets = data;
        trainCheckDatasets( round(n/k)*(iteration-1) + 1 : end , : ) = [];
    end
    
    % find chkData and remove them from dataset
    % what remains are the trnData
    idx = randperm(length(trainCheckDatasets));
    checkIdx = idx(1:round(length(idx)*0.25));
    trainIdx = idx(round(length(idx)*0.25)+1:end);
    
    trainData = trainCheckDatasets(trainIdx,:);
    checkData = trainCheckDatasets(checkIdx,:);
end

