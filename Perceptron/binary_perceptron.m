clc
close all
clear

global N_REPS;
N_REPS = 50;
global T_MAX;
T_MAX = 10^4 ;
global ETA;
ETA = 10^-1;

% Plotting the number of epochs till convergence for P=10, N=10,20,100
P = 10;
for N = [10, 20, 100]
    epochs_arr = zeros(1, N_REPS);
    for n = 1:N_REPS
        [X, y0] = generate_data(N, P);
        [~, converged, epochs] = perceptron(X, y0);
        if converged
            epochs_arr(n) = epochs;
        end
    end
    plot(epochs_arr, '--o')
    disp(mean(epochs_arr(epochs_arr>0)));  
    hold on
end
xlabel('n')
ylabel('epochs')
grid on
hold off
drawnow

% Plotting the probability of convergence as a function of alpha=P/N
for N = [5, 20, 100]
    figure
    P = 1:N/5:3*N;
    alpha = P ./ N;
    prob_converged = zeros(1, length(P));
    e2 = zeros(1, length(P));
    e3 = zeros(1, length(P));
    for i = 1:length(P)
        converged_arr = zeros(1, N_REPS);
        epochs_arr = zeros(1, N_REPS);
        for n = 1:N_REPS
            [X, y0] = generate_data(N, P(i)); 
            [~, converged_arr(n), epochs_arr(n)] = perceptron(X, y0);
        end
        prob_converged(i) = sum(converged_arr == 1) / N_REPS;
        indexes_list = find(converged_arr == 1);
        e2(i) = sum(epochs_arr(indexes_list) <= 10^2) / N_REPS;
        e3(i) = sum(epochs_arr(indexes_list) <= 10^3) / N_REPS;
    end
    plot(alpha, prob_converged, '--o')
    hold on
    plot(alpha, e3, '--o')
    plot(alpha, e2, '--o')
    legend('t_{max}=10^4','t_{max}=10^3','t_{max}=10^2')
    xlabel('$\alpha$', 'interpreter', 'latex')
    ylabel('$Pr$[converged]', 'interpreter', 'latex')
    grid on
    hold off
    drawnow
end

function [X, y0] = generate_data(N, P)
% Generating random dichotomies to the perceptron
    X = randi([0 1], N, P); X(~X) = -1;
    y0 = randi([0 1], 1, P); y0(~y0) = -1;
end

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

