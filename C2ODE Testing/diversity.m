function [p, objF, conV, FES, origin_tracker] = diversity(p, objF, conV, minVar, maxVar, problem, FES, origin_tracker)
    global count;
    [popsize, n] = size(p);
    % Standard C20DE Trigger
    if std(conV) < 1.e-8 && isempty(find(conV == 0))
        p = repmat(minVar, popsize, 1) + rand(popsize, n) .* repmat((maxVar - minVar), popsize, 1);
        [objF, conV] = get_Fitness(p, problem);
        FES = FES + popsize;
        count = count + 1;
        origin_tracker(:) = 1; % Reset everyone as "Restarted"
    end
end