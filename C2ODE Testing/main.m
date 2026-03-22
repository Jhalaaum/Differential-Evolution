clc;
clear all;
close all;
format long;
format compact;
global initial_flag count externalDatabase gen maxGen VAR
initial_flag = 0;
count = 0;
problemSet = 21:28;   
n = 30;              
popsize = 60;
totalFES = 600000;
maxGen = ceil(totalFES / (popsize * 3));

for problemIndex = 1:length(problemSet)
    problem = problemSet(problemIndex);
    fprintf('\n=== Analyzing C20DE Dynamics: Problem %d ===\n', problem);
    
    [lb, ub, ~] = get_Bounds(problem);
    minVar = lb * ones(1, n);
    maxVar = ub * ones(1, n);
    
    % --- LOGGING SETUP ---
    origin_tracker = zeros(popsize, 1); 
    composition_history = zeros(maxGen, 3);
    stagnation_history = zeros(popsize, maxGen);
    ind_stagnation_counters = zeros(popsize, 1);
    
    % Phase Plot Logging: Diversity and Mean Constraint Violation
    diversity_history = zeros(maxGen, 1);
    mean_conV_history = zeros(maxGen, 1);
    
    rng('default'); rng(1); 
    p = repmat(minVar, popsize, 1) + rand(popsize, n) .* repmat((maxVar - minVar), popsize, 1);
    [objF, conV] = get_Fitness(p, problem);
    
    FES = popsize; gen = 1; X = 0; VAR0 = max(conV);
    
    while FES <= totalFES && gen <= maxGen
        cp = (-log(VAR0) - 6) / log(1 - 0.5);
        VAR = (X < 0.5) * (VAR0 * (1 - X)^cp);
        
        % --- Calculate Metrics for Phase Plot ---
        centroid = mean(p, 1);
        diversity_history(gen) = mean(sqrt(sum((p - centroid).^2, 2)));
        mean_conV_history(gen) = mean(conV);
        
        old_objF = objF;
        old_conV = conV;
        
        % 1. Diversity Phase (Restart Logic)
        [p, objF, conV, FES, origin_tracker] = diversity(p, objF, conV, minVar, maxVar, problem, FES, origin_tracker);
        
        % 2. DE Generation
        temp = DEgenerator(p, objF, conV, minVar, maxVar);
        [objFtemp, conVtemp] = get_Fitness(temp, problem);
        FES = FES + popsize * 3;
        
        % 3. Selection
        [trial, objFtrial, conVtrial] = preSelect(temp, objFtemp, conVtemp);
        [p_new, objF_new, conV_new] = epsSelect(p, objF, conV, trial, objFtrial, conVtrial);
        
        % Update Success Tracking and Stagnation Counters
        for i = 1:popsize
            improved = (objF_new(i) < old_objF(i)) || (conV_new(i) < old_conV(i));
            if improved
                ind_stagnation_counters(i) = 0;
                if origin_tracker(i) == 1
                    origin_tracker(i) = 2; % Successfully improved after restart
                end
            else
                ind_stagnation_counters(i) = ind_stagnation_counters(i) + 1;
            end
        end
        
        p = p_new; objF = objF_new; conV = conV_new;
        
        % Log Composition
        composition_history(gen, 1) = sum(origin_tracker == 0) / popsize * 100;
        composition_history(gen, 2) = sum(origin_tracker == 1) / popsize * 100;
        composition_history(gen, 3) = sum(origin_tracker == 2) / popsize * 100;
        stagnation_history(:, gen) = ind_stagnation_counters;
        
        gen = gen + 1; X = X + 1 / maxGen;
    end
    
    % --- PLOT 1: Phase Portrait (Diversity vs Constraint) ---
    figure('Visible', 'off');
    % Color represents the flow of time (Generation)
    scatter(diversity_history(1:gen-1), mean_conV_history(1:gen-1), 15, 1:gen-1, 'filled');
    colormap(jet);
    cb = colorbar;
    ylabel(cb, 'Generation');
    set(gca, 'YScale', 'log'); % Constraint violation is best viewed on log scale
    xlabel('Population Diversity');
    ylabel('Mean Constraint Violation (Log)');
    title(['C20DE Dynamics - Problem ', num2str(problem)]);
    grid on;
    saveas(gcf, sprintf('C20DE_Phase_P%d.png', problem));
    
    % --- PLOT 2: AISR Composition Chart ---
    figure('Visible', 'off');
    area(1:gen-1, composition_history(1:gen-1, :));
    colormap([0.2 0.2 0.8; 0.8 0.2 0.2; 0.2 0.8 0.2]); 
    legend('Original', 'Restarted (Stalled)', 'Restarted (Success)');
    xlabel('Generation'); ylabel('Population %');
    title(['C20DE Composition - Problem ', num2str(problem)]);
    % saveas(gcf, sprintf('C20DE_AISR_P%d.png', problem));
    
    close all;
end