clc;

%load("PIController.mat");
Kp=2.25;
Ki=0.3;
PI=zpk(-Ki/Kp,0,Kp);
Gp = zpk([], [-0.1 -10], 25);

sysOpenLoop = PI*Gp;
sys = feedback(sysOpenLoop,1);
figure(1)
step(sys)
figure(2)
rlocus(sys)
controlSystemDesigner(Gp,PI);
stepinfo(sys)
