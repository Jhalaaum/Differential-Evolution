function [p, objF, conV, FES] = diversity(p, objF, conV, minVar, maxVar, problem, FES)
    global count;
    [popsize, n] = size(p);
    stagnationThreshold = 20;
    stagnationProp = 0.5;

    persistent Gen index Archive_X Archive_Obj Archive_Con PrevObjF stagnatedCount

    if isempty(Gen)
        Gen = 1;
        index = zeros(popsize,1);
        Archive_X = [];
        Archive_Obj = [];
        Archive_Con = [];
        PrevObjF = objF;
        stagnatedCount = [];
    end

    % --- Stagnation check ---
    tol = 1e-12;
    for i = 1:popsize
        if abs(objF(i) - PrevObjF(i)) < tol
            index(i) = index(i) + 1;
            
        else
            index(i) = 0;
        end
    end
    PrevObjF = objF;

    stagnatedMask = index > stagnationThreshold;

    stagnatedCount(Gen) = sum(stagnatedMask);

    % --- Archive update ---
    tempMask = (conV > 0) | stagnatedMask;
    if any(tempMask)
        Archive_X   = [Archive_X; p(tempMask,:)];
        Archive_Obj = [Archive_Obj; objF(tempMask)];
        Archive_Con = [Archive_Con; conV(tempMask)];
        if size(Archive_X,1) > popsize
            Archive_X   = Archive_X(end-popsize+1:end,:);
            Archive_Obj = Archive_Obj(end-popsize+1:end);
            Archive_Con = Archive_Con(end-popsize+1:end);
        end
    end

    % --- Diversity injection ---
    centroid = mean(p,1);
    avgDist = mean(sqrt(sum((p - centroid).^2, 2)));

    if sum(stagnatedMask) > stagnationProp*popsize && avgDist > 1e-12
        for i = find(stagnatedMask)'
            id = randi(size(Archive_X,1));
            p(i,:) = Archive_X(id,:);
            [objF(i), conV(i)] = get_Fitness(p(i,:), problem);
            FES = FES + 1;
            index(i) = 0;
        end
        count=count+1;
    end

    Gen = Gen + 1;

    assignin('base','stagnatedArray',stagnatedCount);
end
