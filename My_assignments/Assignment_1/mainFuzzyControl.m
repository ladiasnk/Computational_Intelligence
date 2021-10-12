clc;%clear;
FLCMatlab = readfis('FLCMATLAB');
FLCSim = FLCMatlab;
fuzzyLogicDesigner(FLCMatlab);
writefis(FLCMatlab,'FLCMATLAB');

