function [p,objF,conV,FES] = diversity(p,objF,conV,minVar,maxVar,problem,FES,ExternalDB)

[popsize,n] = size(p);
infeasibleMask = conV > 0;
fracInfeasible = sum(infeasibleMask) / popsize;

% ---- stagnation detection ----
objStd = std(objF);
if objStd < 1e-8 && fracInfeasible >= 0.8
    preserveFraction = 0.4;           % Total fraction to preserve
    numPreserve = max(1, round(preserveFraction * popsize));
    newPopulation = zeros(popsize, n);

    % ---- 1. Always preserve the best feasible individual ----
    feasibleIdx = find(conV == 0);
    if ~isempty(feasibleIdx)
        [~, bestFeasibleIdx] = min(objF(feasibleIdx));
        bestFeasible = p(feasibleIdx(bestFeasibleIdx), :);
        preserved = bestFeasible;
    else
        preserved = [];
    end

    % ---- 2. Preserve some top stagnated individuals (infeasible) ----
    stagnatedIdx = find(infeasibleMask);
    if ~isempty(stagnatedIdx)
        [~, idx] = sort(objF(stagnatedIdx));
        topStagCount = min(floor(0.1*popsize), length(idx));
        preserved = [preserved; p(stagnatedIdx(idx(1:topStagCount)), :)];
    end

    % ---- 3. Preserve some top non-stagnated individuals (excluding best feasible) ----
    nonStagIdx = setdiff(find(~infeasibleMask), feasibleIdx);
    if ~isempty(nonStagIdx)
        [~, idx] = sort(objF(nonStagIdx));
        topNonStagCount = min(floor(0.1*popsize), length(idx));
        preserved = [preserved; p(nonStagIdx(idx(1:topNonStagCount)), :)];
    end

    % ---- 4. Preserve some individuals from ExternalDB (if available) ----
    if ~isempty(ExternalDB)
        extCount = min(size(ExternalDB,1), floor(0.1*popsize));
        extIdx = randi(size(ExternalDB,1), extCount, 1);
        preserved = [preserved; ExternalDB(extIdx,:)];
    end

    % ---- 5. Trim preserved if exceeds numPreserve ----
    if size(preserved,1) > numPreserve
        preserved = preserved(1:numPreserve,:);
    end

    numPreserveActual = size(preserved,1);
    numRandom = popsize - numPreserveActual;

    % ---- 6. Assign preserved individuals ----
    newPopulation(1:numPreserveActual,:) = preserved;

    % ---- 7. Fill the rest with random individuals ----
    lb = minVar(:)';
    ub = maxVar(:)';
    if numRandom > 0
        newPopulation(numPreserveActual+1:end, :) = rand(numRandom, n) .* (ub - lb) + lb;
    end

    % ---- 8. Update population and recalc fitness ----
    p = newPopulation;
    [objF, conV] = fitness(p, problem);
    FES = FES + numRandom;  % Only count new random evaluations

end
end