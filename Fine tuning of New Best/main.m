clc;
clear all;

format long;
format compact;
global initial_flag
initial_flag = 0;

global externalDatabase;
problemSet = 5:28;   % benchmark functions

n = 30;              % number of dimensions
totalRuns = 0;
feasibleRuns = 0;

% Set parameters depending on dimension
if n == 10
    popsize = 35;
    totalFES = 200000;
elseif n == 30
    popsize = 60;
    totalFES = 600000;
end

totalTime = 25;  % number of runs per problem

global gen maxGen VAR
maxGen = ceil(totalFES / (popsize * 3));
externalDatabase = zeros(maxGen * popsize, n);

% Preallocate arrays for metrics
t1 = zeros(1, length(problemSet));
t2 = zeros(1, length(problemSet));
feasibleRate = zeros(1, length(problemSet));
bestValues = zeros(totalTime, length(problemSet));

% Main loop over problems
for problemIndex = 1:length(problemSet)
    problem = problemSet(problemIndex);
    fprintf('\n=== Problem %d ===\n', problem);

    feasibleCount = 0;
    t1(problemIndex) = 0;
    runTimerTotal = tic;

    [lb, ub, ~] = get_Bounds(problem);
    minVar = lb * ones(1, n);
    maxVar = ub * ones(1, n);

    for run = 1:totalTime
        rng('default'); rng(run);

        % Initialize population
        p = repmat(minVar, popsize, 1) + rand(popsize, n) .* repmat((maxVar - minVar), popsize, 1);
        [objF, conV] = get_Fitness(p, problem);
        FES = popsize;
        gen = 1; X = 0;
        VAR0 = max(conV);
        runTimer = tic;

        % Initialize archive for stagnated vectors
        Archive_TV = [];
        Archive_Val_TV = [];
        Archive_Cons_TV = [];
        index = zeros(maxGen, popsize);

        % --- Main optimization loop ---
        while FES <= totalFES
            cp = (-log(VAR0) - 6) / log(1 - 0.5);
            if X < 0.5
                VAR = VAR0 * (1 - X)^cp;
            else
                VAR = 0;
            end

            % Call updated diversity function based on UDE_IV strategy
            [p, objF, conV, FES] = diversity(p, objF, conV, minVar, maxVar, problem, FES);
    
            % Generate trial vectors and perform DE operations (DEgenerator)
            temp = DEgenerator(p, objF, conV, minVar, maxVar);
            [objFtemp, conVtemp] = get_Fitness(temp, problem);
            FES = FES + popsize * 3;

            % Record t1 at 10,000 evaluations
            if FES >= 10000 && t1(problemIndex) == 0
                t1(problemIndex) = toc(runTimer);
            end

            [trial, objFtrial, conVtrial] = preSelect(temp, objFtemp, conVtemp);
            [p, objF, conV] = epsSelect(p, objF, conV, trial, objFtrial, conVtrial);

            gen = gen + 1;
            X = X + 1 / maxGen;
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
        bestValues(run, problemIndex) = bestSolution;

        fprintf('Run %2d: Best = %.15e | Feasible = %d\n', run, bestSolution, ~isnan(bestSolution));
    end

    % --- Record total runtime for this problem ---
    t2(problemIndex) = toc(runTimerTotal);
    feasibleRate(problemIndex) = feasibleCount / totalTime;

    fprintf('Problem %2d: Feasible Rate = %.2f%% | t1 = %.3f | t2 = %.3f\n', ...
        problem, feasibleRate(problemIndex) * 100, t1(problemIndex), t2(problemIndex));
end

% === Compute complexity metrics ===
T1 = mean(t1);
T2 = mean(t2);
complexityRatio = (T2 - T1) / T1;

fprintf('\n==== FINAL COMPLEXITY RESULTS ====\n');
fprintf('T1 (avg time for 10k evals): %.4f sec\n', T1);
fprintf('T2 (avg total time): %.4f sec\n', T2);
fprintf('Algorithm Complexity Ratio (T2-T1)/T1: %.4f\n', complexityRatio);

fprintf('\nOverall Feasible Rate (all problems): %.2f%%\n', (feasibleRuns / totalRuns) * 100);
fprintf('Total experiment runtime: %.2f sec\n', toc);
