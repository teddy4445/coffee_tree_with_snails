function [t, y] = model_solver()
tspan = [0, 25];
y0_tree_infected = 3;
y0 = [100 - y0_tree_infected, y0_tree_infected, 50];
[t, y] = ode45(@(t,y) model(t,y), tspan, y0);
save model_answer.mat [t, y];
test_plot_model(t, y);
endfunction

function ans = test_plot_model(t, y)
clf;
plot(t,y);
h = legend("T_s", "T_i", "S");
endfunction

function dydt = model(t,y)
a = 0.025;
beta = 0.085;
k = 0.05;
gamma = 0.05;
b = 0.025;
d = 0.15;
dydt = [a * y(1) - beta * y(1) * y(2) + k * y(3) * y(2); beta * y(1) * y(2) - k * y(3) * y(2) - gamma * y(2); b * y(3) * y(2) - d * y(3)];
endfunction

model_solver();
