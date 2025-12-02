function [p,objF,conV,FES] = diversity(p,objF,conV,minVar,maxVar,problem,FES,ExternalDB)

    [popsize,n] = size(p);
    infeasibleMask = conV > 0;
    fracInfeasible = sum(infeasibleMask) / popsize;
 
    % stagnation if constraint violation std is tiny and no feasible individuals (conV == 0)
    if std(conV)<1.e-6 && fracInfeasible >= 0.8

    
        %% Restart Strategy (from second snippet)
        restartFraction = 0.9;
        numToRestart = round(restartFraction * popsize);
        numRemaining = popsize - numToRestart;
    
        externalCount = round(0.1 * popsize);
        topStagnatedCount = round(0.1 * popsize);
        topNonStagnatedCount = round(0.1 * popsize);
        
        % Safety scale if sum > numRemaining
        totalSelected = externalCount + topStagnatedCount + topNonStagnatedCount;
        if totalSelected > numRemaining
            scale = numRemaining / totalSelected;
            externalCount = floor(externalCount * scale);
            topStagnatedCount = floor(topStagnatedCount * scale);
            topNonStagnatedCount = floor(topNonStagnatedCount * scale);
        end
    
        numRandom = popsize - (externalCount + topStagnatedCount + topNonStagnatedCount);
    
        newPopulation = zeros(popsize, n);
    
        % 1. Random new individuals
        lb = minVar(:)';
        ub = maxVar(:)';
        newPopulation(1:numRandom, :) = rand(numRandom, n) .* (ub - lb) + lb;
    
        % 2. Add individuals from ExternalDB if available
        if externalCount > 0 && ~isempty(ExternalDB)
            extSize = size(ExternalDB, 1);
            extIdx = randi(extSize, externalCount, 1);
            newPopulation(numRandom + (1:externalCount), :) = ExternalDB(extIdx, :);
        else
            externalCount = 0; % No external DB individuals added
        end
    
        % 3. Add top stagnated individuals (based on constraint violation)
        infeasibleMask = conV > 0;
        stagnatedIdx = find(infeasibleMask);
        if ~isempty(stagnatedIdx) && topStagnatedCount > 0
            [~, sortedStagIdx] = sort(objF(stagnatedIdx));
            finalStagCount = min(topStagnatedCount, length(sortedStagIdx));
            topStagnated = stagnatedIdx(sortedStagIdx(1:finalStagCount));
            newPopulation(numRandom + externalCount + (1:finalStagCount), :) = p(topStagnated, :);
        else
            finalStagCount = 0;
        end
    
        % 4. Add top non-stagnated individuals
        nonStagnatedIdx = find(~infeasibleMask);
        if ~isempty(nonStagnatedIdx) && topNonStagnatedCount > 0
            top10PercentCount = max(1, ceil(0.3 * length(nonStagnatedIdx)));
            [~, sortedNonStagIdx] = sort(objF(nonStagnatedIdx));
            top10PercentIdx = nonStagnatedIdx(sortedNonStagIdx(1:min(top10PercentCount, end)));
            finalTopCount = min(topNonStagnatedCount, length(top10PercentIdx));
            finalTop = top10PercentIdx(1:finalTopCount);
            newPopulation(numRandom + externalCount + finalStagCount + (1:finalTopCount), :) = p(finalTop, :);
        end
    
        % Assign new population
        p = newPopulation;
    
        % Recalculate fitness and constraints
        [objF, conV] = fitness(p, problem);
        FES = FES + popsize;
    end
end

