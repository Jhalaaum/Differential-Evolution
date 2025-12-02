function [p,objF,conV,FES,prevBest,stagnationCounter] = diversity(...
    p,objF,conV,minVar,maxVar,problem,FES,ExternalDB,prevBest,stagnationCounter)

[popsize,n] = size(p);
infeasibleMask = conV > 0;
fracInfeasible = sum(infeasibleMask)/popsize;

% ---- Identify feasible individuals ----
feasibleIdx = find(conV == 0);

if ~isempty(feasibleIdx)
    [sortedFeasiObj, sortedIdx] = sort(objF(feasibleIdx));
    bestFeasible = p(feasibleIdx(sortedIdx), :);  % sorted feasible
    prevBest = sortedFeasiObj(1);                 % update prevBest
else
    bestFeasible = [];
    prevBest = Inf;                               % no feasible solution
end

% ---- Stagnation detection ----
objStd = std(objF);
if objStd < 1e-4 && fracInfeasible >= 0.8
    stagnationCounter = stagnationCounter + 1;
else
    stagnationCounter = 0;
end

% ---- Partial restart if stagnated ----
if stagnationCounter >= 1
    preserveFraction = 0.4;           % fraction to preserve
    numPreserve = max(1, round(preserveFraction*popsize));
    newPopulation = zeros(popsize,n);

    % ---- 1. Preserve multiple best feasible individuals ----
    preserved = [];
    if ~isempty(bestFeasible)
        numToPreserveFeasi = min(floor(0.2*popsize), size(bestFeasible,1));
        preserved = bestFeasible(1:numToPreserveFeasi, :);
    end

    % ---- 2. Preserve top infeasible (stagnated) individuals ----
    stagnatedIdx = find(infeasibleMask);
    if ~isempty(stagnatedIdx)
        [~, idx] = sort(objF(stagnatedIdx));
        topStagCount = min(floor(0.1*popsize), length(idx));
        preserved = [preserved; p(stagnatedIdx(idx(1:topStagCount)), :)];
    end

    % ---- 3. Preserve top non-stagnated individuals ----
    nonStagIdx = setdiff(find(~infeasibleMask), feasibleIdx);
    if ~isempty(nonStagIdx)
        [~, idx] = sort(objF(nonStagIdx));
        topNonStagCount = min(floor(0.1*popsize), length(idx));
        preserved = [preserved; p(nonStagIdx(idx(1:topNonStagCount)), :)];
    end

    % ---- 4. Preserve individuals from ExternalDB ----
    if ~isempty(ExternalDB)
        extCount = min(size(ExternalDB,1), floor(0.1*popsize));
        extIdx = randi(size(ExternalDB,1), extCount, 1);
        preserved = [preserved; ExternalDB(extIdx,:)];
    end

    % ---- 5. Trim preserved if exceeds numPreserve ----
    if size(preserved,1) > numPreserve
        preserved = preserved(1:numPreserve,:);
    end

    % ---- 6. Ensure at least 1 individual is preserved ----
    if isempty(preserved)
        preserved = rand(1,n).*(maxVar-lb) + minVar;
    end

    numPreserveActual = size(preserved,1);
    numRandom = popsize - numPreserveActual;

    % ---- 7. Assign preserved individuals ----
    newPopulation(1:numPreserveActual,:) = preserved;

    % ---- 8. Fill remaining with random individuals ----
    lb = minVar(:)'; ub = maxVar(:)';
    if numRandom > 0
        newPopulation(numPreserveActual+1:end,:) = rand(numRandom,n) .* (ub-lb) + lb;
    end

    % ---- 9. Update population and fitness ----
    p = newPopulation;
    [objF, conV] = fitness(p, problem);

    % ---- 10. NaN protection ----
    if any(isnan(objF))
        warning('NaN detected in fitness! Replacing with penalty.');
        objF(isnan(objF)) = 1e10;
    end
    if any(isnan(conV))
        conV(isnan(conV)) = 1e10;
    end

    FES = FES + numRandom;   % only count new random evaluations
end

end
