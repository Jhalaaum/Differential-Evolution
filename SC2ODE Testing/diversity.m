function [p, objF, conV, FES, origin_tracker] = diversity(p, objF, conV, minVar, maxVar, problem, FES, origin_tracker)
    global count;
    popsize = size(p,1);
    stagnationThreshold = 20;
    stagnationProp = 0.5;
    persistent Gen index Archive_X Archive_Obj Archive_Con PrevObjF
    if isempty(Gen) || FES <= popsize % Reset persistent variables for new problems
        Gen = 1; index = zeros(popsize,1); Archive_X = []; Archive_Obj = []; Archive_Con = []; PrevObjF = objF;
    end
    
    % Stagnation check
    for i = 1:popsize
        if abs(objF(i) - PrevObjF(i)) < 1e-12, index(i) = index(i) + 1; else, index(i) = 0; end
    end
    PrevObjF = objF;
    stagnatedMask = index > stagnationThreshold;
    
    % Archive update
    tempMask = (conV > 0) | stagnatedMask;
    if any(tempMask)
        Archive_X = [Archive_X; p(tempMask,:)];
        Archive_Obj = [Archive_Obj; objF(tempMask)];
        Archive_Con = [Archive_Con; conV(tempMask)];
        if size(Archive_X,1) > popsize*2
            Archive_X = Archive_X(end-popsize+1:end,:); Archive_Obj = Archive_Obj(end-popsize+1:end); Archive_Con = Archive_Con(end-popsize+1:end);
        end
    end
    
    % Diversity injection with tracking
    centroid = mean(p,1);
    avgDist = mean(sqrt(sum((p - centroid).^2, 2)));
    if sum(stagnatedMask) > stagnationProp*popsize && avgDist > 1e-12 && ~isempty(Archive_X)
        for i = find(stagnatedMask)'
            id = randi(size(Archive_X,1));
            p(i,:) = Archive_X(id,:);
            [objF(i), conV(i)] = get_Fitness(p(i,:), problem);
            FES = FES + 1;
            index(i) = 0;
            origin_tracker(i) = 1; % Tag as "Fresh Archive Recruit"
        end
        count = count + 1;
    end
    Gen = Gen + 1;
end