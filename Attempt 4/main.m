%% MAIN SCRIPT FOR C2ODE WITH STAGNATION-AWARE PARTIAL RESTART
clc;
clear all;

format long;
format compact;

global externalDatabase;
problemSet = 11:11;   % benchmark problems

n = 30;              % number of dimensions
totalRuns = 0;
feasibleRuns = 0;

% Set population size and total FES depending on dimensions
if n == 10
    popsize = 35;
    totalFES = 200000;
elseif n == 30
    popsize = 60;
    totalFES = 600000;
end

totalTime = 25;      % number of runs per problem

global gen maxGen VAR
maxGen = ceil(totalFES / (popsize * 3));
externalDatabase = zeros(maxGen * popsize, n);

% Preallocate arrays for metrics
t1 = zeros(1, length(problemSet));           % time for 10,000 evaluations
t2 = zeros(1, length(problemSet));           % total time
feasibleRate = zeros(1, length(problemSet));
bestValues = zeros(totalTime, length(problemSet));  % store best solution per run

for problemIndex = 1:length(problemSet)
    problem = problemSet(problemIndex);
    fprintf('\n=== Problem %d ===\n', problem);

    feasibleCount = 0;
    tic;  % start timer for total time
    t1(problemIndex) = 0;

    for run = 1:totalTime
        % --- Standardised RNG ---
        rng('default'); rng(run);

        % --- Define bounds for each problem ---
        switch problem
            case 1,  minVar=zeros(1,n);      maxVar=10*ones(1,n);
            case 2,  minVar=-5.12*ones(1,n); maxVar=5.12*ones(1,n);
            case 3,  minVar=-1000*ones(1,n); maxVar=1000*ones(1,n);
            case 4,  minVar=-50*ones(1,n);   maxVar=50*ones(1,n);
            case 5,  minVar=-600*ones(1,n);  maxVar=600*ones(1,n);
            case 6,  minVar=-600*ones(1,n);  maxVar=600*ones(1,n);
            case 7,  minVar=-140*ones(1,n);  maxVar=140*ones(1,n);
            case 8,  minVar=-140*ones(1,n);  maxVar=140*ones(1,n);
            case 9,  minVar=-500*ones(1,n);  maxVar=500*ones(1,n);
            case 10, minVar=-500*ones(1,n);  maxVar=500*ones(1,n);
            case 11, minVar=-100*ones(1,n);  maxVar=100*ones(1,n);
            case 12, minVar=-1000*ones(1,n); maxVar=1000*ones(1,n);
            case 13, minVar=-500*ones(1,n);  maxVar=500*ones(1,n);
            case 14, minVar=-1000*ones(1,n); maxVar=1000*ones(1,n);
            case 15, minVar=-1000*ones(1,n); maxVar=1000*ones(1,n);
            case 16, minVar=-10*ones(1,n);   maxVar=10*ones(1,n);
            case 17, minVar=-10*ones(1,n);   maxVar=10*ones(1,n);
            case 18, minVar=-50*ones(1,n);   maxVar=50*ones(1,n);
        end

        % --- Initialize population ---
        p = minVar + rand(popsize,n).*(maxVar-minVar);
        [objF, conV] = fitness(p, problem);
        FES = popsize;
        gen = 1; X = 0; VAR0 = max(conV);

        % --- Initialize stagnation tracking ---
        prevBest = Inf;
        stagnationCounter = 0;

        runTimer = tic;

        % --- Main optimization loop ---
        while FES <= totalFES
            % --- Update VAR (for C2ODE) ---
            cp = (-log(VAR0)-6)/log(1-0.5);
            if X < 0.5, VAR = VAR0*(1-X)^cp; else, VAR = 0; end

            % --- Diversity / partial restart step ---
            [p, objF, conV, FES, prevBest, stagnationCounter] = diversity( ...
                p, objF, conV, minVar, maxVar, problem, FES, externalDatabase, prevBest, stagnationCounter);

            % --- Differential Evolution generation ---
            temp = DEgenerator(p, objF, conV, minVar, maxVar);
            [objFtemp, conVtemp] = fitness(temp, problem);
            FES = FES + popsize*3;

            % --- Record t1 when 10,000 evaluations reached ---
            if FES >= 10000 && t1(problemIndex) == 0
                t1(problemIndex) = toc(runTimer);
            end

            % --- Preselection and epsilon selection ---
            [trial, objFtrial, conVtrial] = preSelect(temp, objFtemp, conVtemp);
            [p, objF, conV] = epsSelect(p, objF, conV, trial, objFtrial, conVtrial);

            gen = gen + 1;
            X = X + 1/maxGen;
        end

        % --- Feasibility check ---
        feasiIndex = find(conV == 0);
        if ~isempty(feasiIndex)
            feasibleCount = feasibleCount + 1;
            [sortedFeasiVec, ~] = sort(objF(feasiIndex));
            bestSolution = sortedFeasiVec(1);
        else
            bestSolution = NaN;
        end

        totalRuns = totalRuns + 1;
        if ~isnan(bestSolution), feasibleRuns = feasibleRuns + 1; end
        fprintf('Run %2d: Best = %.15d | Feasible = %d\n', run, bestSolution, ~isnan(bestSolution));
        bestValues(run, problemIndex) = bestSolution;
    end

    % --- Record total runtime for this problem ---
    t2(problemIndex) = toc;
    feasibleRate(problemIndex) = feasibleCount / totalTime;
end

% === Compute complexity metrics ===
T1 = mean(t1);
T2 = mean(t2);
complexityRatio = (T2-T1)/T1;

fprintf('\n==== FINAL COMPLEXITY RESULTS ====\n');
fprintf('T1 (avg time for 10k evals): %.4f sec\n', T1);
fprintf('T2 (avg total time): %.4f sec\n', T2);
fprintf('Algorithm Complexity Ratio (T2-T1)/T1: %.4f\n', complexityRatio);

fprintf('\n==== FEASIBILITY RESULTS ====\n');
for i = 1:length(problemSet)
    fprintf('Problem %2d: Feasible Rate = %.2f%% | t1 = %.3f | t2 = %.3f\n', ...
        i, feasibleRate(i)*100, t1(i), t2(i));
end

fprintf('\nOverall Feasible Rate (all problems): %.2f%%\n', (feasibleRuns/totalRuns)*100);
fprintf('Total experiment runtime: %.2f sec\n', toc);
