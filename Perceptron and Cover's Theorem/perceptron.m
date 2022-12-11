clc
close all
clear

global N_REPS;
N_REPS = 50;
global T_MAX;
T_MAX = 10^4 ;
global ETA;
ETA = 10^-1;

function [w, converged, epochs] = perceptron(X, y0)
% Implementing a simple binary-perceptron w/ a sign activation function
    [N, ~] = size(X);
    w = rand(N, 1); 
    global T_MAX;
    global ETA;
    epochs = 0;
    converged = false;
    while (epochs < T_MAX) && ~converged
        converged = true;
        res = (w' * X) .* y0; 
        violation_indexes = find(res < 0);
        for i = violation_indexes    
            w = w + ETA .* X(:, i) * y0(i);
            % plot(w, '.:'); title(epochs); drawnow 
            converged = false;
        end   
        epochs = epochs+1;
    end
end 