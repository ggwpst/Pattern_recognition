
input=[]
target=[];
out=[];
s=[];
y=[];
x1=[];
x2=[];
x3=[];
x4=[];

%題目
for i=1:1:600
    x1=rand;
    x2=rand;
    x3=rand;
    x4=rand;
    s=[x1, x2, x3, x4];
    input=[input;s];
    y=0.8*x1*x2*x3*x4+x1.^2+x2.^2+x3.^3+x4.^2+x1+x2*0.7-x2.^2*x3.^2+0.5*x1*x4.^2+x4*x2.^3+(-x1)*x2+(x1*x2*x3*x4).^3+(x1-x2+x3-x4)+(x1*x4)-(x2*x3)-2;
    target=[target;y]
end

outnet=[];

%initialize the weight matrix
outputmatrix = zeros(35,1);
for i=1:1:35
    for j=1:1:1
        outputmatrix(i,j)=rand;
    end
end

hiddenmatrix=zeros(4,35);
for i=1:1:4
    for j=1:1:35
        hiddenmatrix(i,j)=rand;
    end
end

trmse = [];
srmse = [];
outputupdate = [];
hiddenupdate = [];
for epoch=1:1:100
    for i=1:1:400
        logsigmatrix = logsig(input(i, :) * hiddenmatrix);
        purelindata = logsigmatrix * outputmatrix;
        deltadata = target(i) - purelindata;
        outputupdate = deltadata * logsigmatrix;
        outputupdate = outputupdate';
        hiddenupdate = deltadata * outputmatrix .* dlogsig(logsigmatrix, logsigmatrix)' * input(i, :);
        hiddenupdate = hiddenupdate'; 
        hiddenmatrix = hiddenmatrix + 0.1 * hiddenupdate;
        outputmatrix = outputmatrix + 0.1 * outputupdate;
    end

    training = 0;
    simulation = 0;
    for i = 1:1:400
        logsigmatrix = logsig(input(i, :) * hiddenmatrix); 
        purelindata = purelin(logsigmatrix * outputmatrix);
        deltadata = target(i) - purelindata;
        training = training + deltadata .* deltadata;
    end
    training = sqrt(training / 400);
    trmse = [trmse, training];
    for simulationsamples = 401:1:600
        logsigmatrix = logsig(input(simulationsamples, :) * hiddenmatrix); 
        purelindata = logsigmatrix * outputmatrix;
        deltadata = target(simulationsamples) - purelindata;
        simulation = simulation + deltadata ^ 2;
    end
    simulation = sqrt(simulation / 200);
    srmse = [srmse, simulation];
end    

epo = [];
samples = [];
predict = [];
for i = 1:1:100
    epo = [epo, i];
end
for i = 1:1:600
    samples = [samples, i];
    logsigmatrix = logsig(input(i, :) * hiddenmatrix); 
    purelindata = logsigmatrix * outputmatrix;
    predict = [predict, purelindata];
end
figure;
plot(epo, trmse, epo, srmse);
xlabel('epoch');
ylabel('Loss');
legend('Training', 'Simulation');
figure;
plot(samples, target', samples, predict);
xlabel('samples');
ylabel('value');
legend('Function', 'Simulation');