%% the code for IEEE CEC2006 constrained benchmark
%*********************************************************************************
%  Author: Bing-Chuan Wang and Yong Wang
%  Last Edited: 28/02/2018
%  Email: bingcwang3-c@my.cityu.edu.hk and ywang@csu.edu.cn
%  Reference: Composite Differential Evolution for Constrained Evolutionary Optimization
%*********************************************************************************
%% introduction to the m files
%**********************************************************************************
%  mian.m is the main procedure
%  DEgenerator.m is the generator function based on differential evolution  
%  diversity.m is the restart strategy
%  epsSelect.m is the selection function based on the epsilon constraint-handling technique
%  preSelect.m is the preselection fucntion based on the feasibility rule
%  fitness.m is the fitness function
%%
clc;
% clear all;

format long;
format compact;
tic;

% problem set of 18 benchmark functions to be tested
problemSet=[1:19 21 23 24];


% the maximum number of function evaluations
totalFES=500000;



% the size of evolving population

if totalFES==500000
    popsize=60;
elseif totalFES==240000
    popsize=50;
end


% the value which represents the number of total runs
 totalTime=1;
 
global gen maxGen VAR 

% choose the function to be tested
for problemIndex= 1:22
    
    problem = problemSet(problemIndex);
    
    % the value used to record the number of runs
    time=1;
    
   
    
    % the array used to record the best solution of each run
    minValue=[];
    
    while time <= totalTime
        switch problem
            case 1
                
                % lu: define the upper and lower bounds of the variables
                lu = [0 0 0 0 0 0 0 0 0 0 0 0 0;  1 1 1 1 1 1 1 1 1 100 100 100 1];
                
                % n: define the dimension of the problem
                n = 13; aaa = []; %\\\\-15\
                
            case 2
                
                lu = [zeros(1, 20); 10 * ones(1, 20)];
                n = 20; aaa = []; %\\\-0.803619\
                
            case 3
                
                lu = [zeros(1, 10); ones(1, 10)];
                n = 10; aaa = []; %\\\-1\
                
            case 4
                
                lu = [78 33 27 27 27; 102 45 45 45 45];
                n = 5; aaa = []; %-30665.539\
                
            case 5
                
                lu = [0 0 -0.55 -0.55; 1200 1200 0.55 0.55];
                n = 4; aaa = []; %\\5126.4981\
                
            case 6
                
                lu = [13 0; 100 100];
                n = 2; aaa = []; %\\-6961.81388\
                
            case 7
                
                lu = [-10 * ones(1, 10); 10 * ones(1, 10)];
                n = 10; aaa = []; %\\24.306\
                
            case 8
                
                lu = [0 0; 10 10];
                n = 2; aaa = []; %\0.095825\
                
            case 9
                
                lu = [-10 -10 -10 -10 -10 -10 -10; 10 10 10 10 10 10 10];
                n = 7; aaa = []; %\680.6300573\
                
            case 10
                
                lu = [100 1000 1000 10 10 10 10 10; 10000 10000 10000 1000 1000 1000 1000 1000];
                n = 8; aaa = []; %\7049.2480\
                
            case 11
                
                lu = [-1 -1; 1 1];
                n = 2; aaa = []; %\\0.75\
                
            case 12
                
                lu = [0 0 0; 10 10 10];
                n = 3; %\\-1\
                l = 1;
                for i = 1:9
                    for j = 1:9
                        for k = 1:9
                            aaa(l, :) = [i j k];
                            l = l+1;
                        end
                    end
                end
                
            case 13
                
                lu = [-2.3 -2.3 -3.2 -3.2 -3.2; 2.3 2.3 3.2 3.2 3.2];
                n = 5; aaa = []; %\0.0539498\
                
            case 14
                
                lu = [zeros(1, 10);  10 * ones(1, 10)];
                n = 10; aaa = []; %\-47.7648884595\
                
            case 15
                
                lu = [zeros(1, 3); 10 * ones(1, 3)];
                n = 3; aaa = []; %\961.7150222899\
                
            case 16
                
                lu = [704.4148 68.6 0 193 25; 906.3855 288.88 134.75 287.0966 84.1988];
                n = 5; aaa = []; %\-1.9051552586\
                
            case 17
                
                lu = [0 0 340 340 -1000 0;  400 1000 420 420 1000 0.5236];
                n = 6; aaa = []; %\8853.5396748064\
                
            case 18
                
                lu = [-10 -10 -10 -10 -10 -10 -10 -10 0;  10 10 10 10 10 10 10 10 20];
                n = 9; aaa = []; %\-0.8660254038\
                
            case 19
                
                lu = [zeros(1, 15);  10 * ones(1, 15)];
                n = 15; aaa = []; %\32.6555929502\
                
            case 20
                
                lu = [zeros(1, 24); 10 * ones(1, 24)];
                n = 24; aaa = []; %\0.2049794002\
                
            case 21
                
                lu = [0 0 0 100 6.3 5.9 4.5; 1000 40 40 300 6.7 6.4 6.25];
                n = 7; aaa = []; %\193.7245100700\
                
            case 22
                
                lu = [0 0 0 0 0 0 0 100 100 100.01 100 100 0 0 0  0.01 0.01 -4.7 -4.7 -4.7 -4.7 -4.7; 20000 10^6 10^6 10^6 4 * 10^7 ...
                    4 * 10^7 4 * 10^7 299.99 399.99 300 400 600 500 500 500 300 400 6.25 6.25 6.25 6.25 6.25];
                n = 22; aaa = []; %\236.4309755040\
                
            case 23
                
                lu = [0 0 0 0 0 0 0 0 0.01;  300 300 100 200 100 300 100 200 0.03];
                n = 9; aaa = []; %\-400.0551000000\
                
            case 24
                
                lu = [0 0;  3 4];
                n = 2; aaa = []; %\-5.5080132716\
        end
        maxGen=ceil(totalFES/(popsize*3));
        % the lower boundary
        minVar=lu(1,:);
        
        % the upper boundary
        maxVar=lu(2,:);
        
       rand('seed',sum(100*clock));
        
        % initialize the population
        p=repmat(minVar,popsize,1)+rand(popsize,n).*repmat((maxVar-minVar),popsize,1);
        
        % evaluate teh population
        [objF, conV]=fitness(p, problem,aaa);
        
        
        FES=popsize;
        
        gen=1;    X=0;
        
        VAR0=max(conV);         
        cp=(-log(VAR0)-6)/log(1-0.5);
        
        while FES<=totalFES
            
            % adjusting the threshold
            if X < 0.5
                VAR=VAR0*(1-X)^cp;
            else
                VAR=0;
            end
      
            [p,objF,conV,FES]=diversity(p,objF,conV,minVar,maxVar,problem,FES,aaa);
         
            % generating next generation
             temp=DEgenerator(p,objF,conV,minVar,maxVar);
             [objFtemp,conVtemp]=fitness(temp,problem,aaa);
             FES=FES+popsize*3;
            
            [trial, objFtrial, conVtrial]=preSelect(temp,objFtemp,conVtemp);
            
            % slecting next generation and updating archive
            [p,objF,conV]=epsSelect(p,objF,conV,trial,objFtrial,conVtrial);    
                              
            gen=gen+1;
            X=X+1/maxGen;
                        
        end
        
        % modify the value of "time"
        time=time+1;
        
        % calculate the best solution of each run
        % calculate the index of feasible solutions
        feasiIndex=find(conV==0);
        if problem==20
            [~,r]=min(conV);
             bestSolution=objF(r);
        else
            if ~isempty(feasiIndex)
                
                % sort the feasible solutions according to their objective
                % function values
                [sortedFeasiVec,~]=sort(objF(feasiIndex));
                
                % the best solution is the feasible solution with minimum
                % objective function value
                bestSolution=sortedFeasiVec(1);
                
            else
                
                % if the population is infeasible, the best solution is
                % assigned the value of NaN
                bestSolution=NaN;
                
            end
        end
        % record the best solution of each run
        minValue=[minValue bestSolution];
        [bestSolution]
    end
    
    % calculate the best value, mean value, worst value and standard error
    % of the array of minValue
    statisticValue=[min(minValue) mean(minValue) max(minValue) std(minValue)]

end
% csvwrite('data2006.csv',dataRecord);
% csvwrite('allData2006.csv',allData);
run_time = toc