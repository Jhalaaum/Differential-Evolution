function [p, objF, conV, FES, prevBest, stagnationCounter] = diversity(...
    p, objF, conV, minVar, maxVar, problem, FES, ExternalDB, prevBest, stagnationCounter)

[popsize, n] = size(p);

% ---- Parameters ----
tolerance = 1e-2;         % minimal significant improvement for feasible best
divThreshold = 1.e-8;       % normalized population diversity threshold
preserveFraction = 0.5;    % fraction of population to preserve
maxExternalFraction = 0.3; % fraction of population from ExternalDB
stagnationCounter = 0;

lb = minVar(:)'; ub = maxVar(:)';

% ---- 1. Best feasible solution ----
feasibleIdx = find(conV == 0);
if ~isempty(feasibleIdx)
    bestFeasible = min(objF(feasibleIdx));
else
    bestFeasible = Inf;
end

% ---- 2. Check improvement over previous best feasible ----
if bestFeasible >= prevBest - tolerance
    stagnationCounter = stagnationCounter + 1;
else
    stagnationCounter = 0;
    prevBest = bestFeasible;
end

% ---- 3. Population diversity ----
diversity = mean(std(p) ./ (ub - lb));

% ---- 4. Fraction of feasible individuals ----
fracFeasible = sum(conV == 0) / popsize;

% ---- 5. Trigger partial restart ----
if diversity < divThreshold
    stagnationLimit = 10
    numPreserve = max(1, round(preserveFraction * popsize));
    newPopulation = zeros(popsize, n);

    % ---- 5a. Preserve top feasible individuals ----
    preserved = [];
    if ~isempty(feasibleIdx)
        [~, sortIdx] = sort(objF(feasibleIdx));
        topCount = min(numPreserve, length(sortIdx));
        preserved = p(feasibleIdx(sortIdx(1:topCount)), :);
    end

    % ---- 5b. Fill up preserved from top infeasible if needed ----
    if size(preserved,1) < numPreserve
        infeasibleIdx = setdiff(1:popsize, feasibleIdx);
        if ~isempty(infeasibleIdx)
            [~, sortIdx] = sort(objF(infeasibleIdx));
            topInfeasCount = min(numPreserve - size(preserved,1), length(sortIdx));
            preserved = [preserved; p(infeasibleIdx(sortIdx(1:topInfeasCount)), :)];
        end
    end

    % ---- 5c. Include some individuals from ExternalDB ----
    if ~isempty(ExternalDB)
        extCount = min(size(ExternalDB,1), floor(maxExternalFraction*popsize));
        extIdx = randi(size(ExternalDB,1), extCount, 1);
        preserved = [preserved; ExternalDB(extIdx,:)];

        % Trim to numPreserve if exceeds
        if size(preserved,1) > numPreserve
            preserved = preserved(1:numPreserve,:);
        end
    end

    numPreserveActual = size(preserved,1);
    numRandom = popsize - numPreserveActual;

    % ---- 5d. Assign preserved and random individuals ----
    newPopulation(1:numPreserveActual,:) = preserved;
    if numRandom > 0
        newPopulation(numPreserveActual+1:end,:) = rand(numRandom, n) .* (ub - lb) + lb;
    end

    % ---- 5e. Update population ----
    p = newPopulation;
    [objF, conV] = get_Fitness(p, problem);
    FES = FES + numRandom; % only count new random evaluations
end

end
