% run_simscape_validation.m
% =========================================================================
% Runs 25 points from the 100-sample LHS through the Simscape switching
% model and saves the validation dataset used for the fidelity-gap analysis.
% =========================================================================

%% Configuration
this_dir = fileparts(mfilename('fullpath'));
src_dir = fileparts(this_dir);
repo_root = fileparts(src_dir);

modelName   = 'boost_switching';
model_file = fullfile(this_dir, [modelName '.slx']);
hifi_file = fullfile(repo_root, 'data', 'raw', 'simulation_results_hifi.csv');
output_file = fullfile(repo_root, 'data', 'raw', 'simscape_validation_results.csv');

% Fixed component values (same as ODE model)
L = 100e-6;
C = 100e-6;

% Select every 4th row -> 25 of 100 points
subset_idx = 1:4:100;

%% Block name references (from find_system)
blk_vsrc  = [modelName '/DC Voltage Source'];
blk_res   = [modelName '/Resistor'];
blk_pulse = [modelName '/Pulse' newline 'Generator'];

%% Load the averaged-model dataset
hifi = readtable(hifi_file);
sub  = hifi(subset_idx, :);
N    = height(sub);

fprintf('=== Simscape Validation Campaign ===\n');
fprintf('Points to run: %d\n', N);
fprintf('Model: %s\n\n', modelName);

%% Verify block access
load_system(model_file);

% Quick check: read current parameter values to confirm names are right
try
    fprintf('DC Voltage Source DC = %s\n', get_param(blk_vsrc, 'DC'));
    fprintf('Resistor R = %s\n', get_param(blk_res, 'R'));
    fprintf('Pulse Generator Period = %s\n', get_param(blk_pulse, 'Period'));
    fprintf('Pulse Generator PulseWidth = %s\n', get_param(blk_pulse, 'PulseWidth'));
    fprintf('Block names verified OK.\n\n');
catch ME
    fprintf('ERROR accessing block parameters: %s\n', ME.message);
    fprintf('Check block names with find_system.\n');
    return;
end

%% Preallocate results table
res = table( ...
    zeros(N,1), zeros(N,1), zeros(N,1), zeros(N,1), ...
    zeros(N,1), zeros(N,1), ...
    zeros(N,1), zeros(N,1), ...
    zeros(N,1), zeros(N,1), ...
    zeros(N,1), zeros(N,1), ...
    zeros(N,1), zeros(N,1), ...
    zeros(N,1), ...
    repmat({''},N,1), ...
    'VariableNames', { ...
        'D','V_in','R','f_sw', ...
        'ode_v_out_mean','simscape_v_out_mean', ...
        'ode_i_l_mean','simscape_i_l_mean', ...
        'ode_v_out_ripple','simscape_v_out_ripple', ...
        'ode_i_l_ripple','simscape_i_l_ripple', ...
        'ode_efficiency','simscape_efficiency', ...
        'wall_time_s','status'});

%% Run campaign
total_tic = tic;

for i = 1:N
    fprintf('Point %2d/%d  D=%.3f  V_in=%.1f  R=%.1f  f_sw=%.0f Hz ... ', ...
        i, N, sub.D(i), sub.V_in(i), sub.R(i), sub.f_sw(i));

    point_tic = tic;

    % Extract parameters
    D     = sub.D(i);
    v_in  = sub.V_in(i);
    R     = sub.R(i);
    f_sw  = sub.f_sw(i);
    T_sw  = 1 / f_sw;

    % Store inputs
    res.D(i) = D;
    res.V_in(i) = v_in;
    res.R(i) = R;
    res.f_sw(i) = f_sw;
    res.ode_v_out_mean(i) = sub.v_out_mean(i);
    res.ode_i_l_mean(i) = sub.i_l_mean(i);
    res.ode_v_out_ripple(i) = sub.v_out_ripple(i);
    res.ode_i_l_ripple(i) = sub.i_l_ripple(i);
    res.ode_efficiency(i) = sub.efficiency(i);

    % Set block parameters via set_param
    set_param(blk_vsrc,  'DC',         num2str(v_in, '%.6f'));
    set_param(blk_res,   'R',          num2str(R, '%.6f'));
    set_param(blk_pulse, 'Period',     num2str(T_sw, '%.10f'));
    set_param(blk_pulse, 'PulseWidth', num2str(D * 100, '%.4f'));

    % Simulation time: enough for steady state
    tau   = R * C;
    t_end = max(100 * tau, 500 * T_sw);
    t_end = max(t_end, 0.01);
    set_param(modelName, 'StopTime', num2str(t_end, '%.6f'));

    try
        % Run simulation
        simOut = sim(modelName);

        % Extract timeseries
        v_out_data = simOut.get('v_out_ts');
        i_l_data   = simOut.get('i_l_ts');

        t     = v_out_data.Time;
        v_out = squeeze(v_out_data.Data);
        i_l   = squeeze(i_l_data.Data);

        % Steady-state: last 10 switching cycles
        t_ss_start = t(end) - 10 * T_sw;
        ss = t >= t_ss_start;

        if sum(ss) < 10
            error('Not enough steady-state points (%d)', sum(ss));
        end

        v_ss = v_out(ss);
        i_ss = i_l(ss);

        % Compute outputs (same definitions as ODE model)
        v_out_mean   = mean(v_ss);
        i_l_mean     = mean(i_ss);
        v_out_ripple = max(v_ss) - min(v_ss);
        i_l_ripple   = max(i_ss) - min(i_ss);

        P_out = v_out_mean^2 / R;
        P_in  = v_in * i_l_mean;

        if P_in > 0
            efficiency = min(P_out / P_in, 0.99);
        else
            efficiency = NaN;
        end

        wall_time = toc(point_tic);

        res.simscape_v_out_mean(i) = v_out_mean;
        res.simscape_i_l_mean(i) = i_l_mean;
        res.simscape_v_out_ripple(i) = v_out_ripple;
        res.simscape_i_l_ripple(i) = i_l_ripple;
        res.simscape_efficiency(i) = efficiency;
        res.wall_time_s(i) = wall_time;
        res.status{i} = 'OK';

        fprintf('V_out=%.2fV  eta=%.3f  (%.1fs)\n', v_out_mean, efficiency, wall_time);

    catch ME
        wall_time = toc(point_tic);
        fprintf('FAILED: %s\n', ME.message);

        res.simscape_v_out_mean(i) = NaN;
        res.simscape_i_l_mean(i) = NaN;
        res.simscape_v_out_ripple(i) = NaN;
        res.simscape_i_l_ripple(i) = NaN;
        res.simscape_efficiency(i) = NaN;
        res.wall_time_s(i) = wall_time;
        res.status{i} = ME.message;
    end
end

total_time = toc(total_tic);

%% Save results
writetable(res, output_file);

%% Summary
n_ok   = sum(strcmp(res.status, 'OK'));
n_fail = N - n_ok;
fprintf('\n=== Campaign Complete ===\n');
fprintf('Passed: %d/%d\n', n_ok, N);
fprintf('Failed: %d/%d\n', n_fail, N);
fprintf('Total time: %.1f s  (mean %.1f s/point)\n', total_time, total_time/N);
fprintf('Results saved to: %s\n', output_file);

%% Close model
close_system(modelName, 0);
