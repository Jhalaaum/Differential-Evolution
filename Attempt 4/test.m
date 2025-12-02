clear all;
clc;

% --- Setup problem and population ---
problem.lb = -5 * ones(20,1);
problem.ub = 5 * ones(20,1);
problem.type = 1;  % Using problem 1 from your fitness function

popsize = 20;
n = 20;

% Create initial population within bounds
p = rand(popsize, n) .* (problem.ub' - problem.lb') + problem.lb';

% Evaluate fitness and constraints
[objF, conV] = fitness(p, problem.type);

% Force stagnation by setting constraint violations high and uniform
conV = ones(popsize,1) * 10;  % all infeasible and identical constraint violation

minVar = -inf;
maxVar = inf;

FES = 0;

% External database with some random individuals
ExternalDB = rand(5, n) .* (problem.ub' - problem.lb') + problem.lb';

% --- Call diversity function ---
[pNew, objFNew, conVNew, FESNew] = diversity(p, objF, conV, minVar, maxVar, problem, FES, ExternalDB);

% --- Print results ---
fprintf('\n--- Diversity Function Results ---\n');

% Check if population changed
if isequal(p, pNew)
    fprintf('Population did NOT change.\n');
else
    fprintf('Population was UPDATED by diversity restart.\n');
end

% Print first few individuals before and after restart
fprintf('\nFirst 3 individuals before restart:\n');
disp(p(1:3,:));

fprintf('First 3 individuals after restart:\n');
disp(pNew(1:3,:));

% FES count increment
fprintf('FES before: %d\n', FES);
fprintf('FES after:  %d\n', FESNew);

fprintf('Difference in FES (should equal population size %d): %d\n', popsize, FESNew - FES);

