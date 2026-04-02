function build_simple_boost_model()
% BUILD_SIMPLE_BOOST_MODEL
% Creates a simplified boost converter model using averaged equations
%
% This is a simplified version that doesn't require complex Simscape blocks.
% It uses a mathematical model of the boost converter for faster simulation
% and easier Python integration.
%
% Usage:
%   build_simple_boost_model()
%
% The model will be saved as 'boost_converter_simple.slx'

%% Model Name
modelName = 'boost_converter_simple';

%% Close and delete existing model if it exists
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
if exist([modelName '.slx'], 'file')
    delete([modelName '.slx']);
end

%% Create new model
new_system(modelName);
open_system(modelName);

%% Add MATLAB Function block with boost converter equations
% Position for main function block
pos_fcn = [200, 100, 400, 200];

% Add MATLAB Function block
add_block('simulink/User-Defined Functions/MATLAB Function', ...
    [modelName '/Boost_Converter'], 'Position', pos_fcn);

% Set the function code
matlabFunctionBlock = [modelName '/Boost_Converter'];

% Get the chart object
chart = find(slroot, '-isa', 'Stateflow.EMChart', 'Path', matlabFunctionBlock);

if ~isempty(chart)
    % Define the boost converter dynamics as a MATLAB function
    boostConverterCode = sprintf([...
        'function [v_out, i_l, v_out_ripple, i_l_ripple] = fcn(duty_cycle, v_in, load_r, L, C, f_sw)\n' ...
        '%% Boost Converter Averaged Model\n' ...
        '% Calculates steady-state outputs using averaged equations\n\n' ...
        '%% Steady-state calculations\n' ...
        '% Ideal boost converter relationship\n' ...
        'v_out = v_in / (1 - duty_cycle);\n\n' ...
        '% Output current\n' ...
        'i_out = v_out / load_r;\n\n' ...
        '% Input (inductor) current\n' ...
        'i_l = i_out / (1 - duty_cycle);\n\n' ...
        '%% Ripple calculations\n' ...
        'T_sw = 1 / f_sw;  % Switching period\n\n' ...
        '% Inductor current ripple (peak-to-peak)\n' ...
        'delta_i_L = (v_in * duty_cycle * T_sw) / L;\n' ...
        'i_l_ripple = delta_i_L;\n\n' ...
        '% Output voltage ripple (peak-to-peak)\n' ...
        'delta_v_C = (i_out * duty_cycle * T_sw) / C;\n' ...
        'v_out_ripple = delta_v_C;\n\n' ...
        '% Add some realistic variations (±2%%)\n' ...
        'v_out = v_out * (1 + 0.02 * randn());\n' ...
        'i_l = i_l * (1 + 0.02 * randn());\n\n' ...
        'end\n']);

    % Set the function code
    chart.Script = boostConverterCode;
end

%% Add input constant blocks
input_y_start = 80;
input_spacing = 50;

inputs = {'duty_cycle', 'v_in', 'load_r', 'L', 'C', 'f_sw'};
for i = 1:length(inputs)
    y_pos = input_y_start + (i-1) * input_spacing;
    blockName = [modelName '/' inputs{i} '_in'];
    add_block('simulink/Sources/Constant', blockName, ...
        'Value', inputs{i}, ...
        'Position', [50, y_pos, 120, y_pos+30]);

    % Connect to function block
    add_line(modelName, [inputs{i} '_in/1'], ['Boost_Converter/' num2str(i)], ...
        'autorouting', 'on');
end

%% Add output To Workspace blocks
outputs = {'v_out', 'i_l', 'v_out_ripple', 'i_l_ripple'};
output_y_start = 100;
output_spacing = 50;

for i = 1:length(outputs)
    y_pos = output_y_start + (i-1) * output_spacing;
    blockName = [modelName '/' outputs{i} '_log'];
    add_block('simulink/Sinks/To Workspace', blockName, ...
        'VariableName', outputs{i}, ...
        'SaveFormat', 'Array', ...
        'Position', [500, y_pos, 600, y_pos+30]);

    % Connect from function block
    add_line(modelName, ['Boost_Converter/' num2str(i)], [outputs{i} '_log/1'], ...
        'autorouting', 'on');
end

%% Add display blocks for visualization
for i = 1:length(outputs)
    y_pos = output_y_start + (i-1) * output_spacing;
    blockName = [modelName '/' outputs{i} '_display'];
    add_block('simulink/Sinks/Display', blockName, ...
        'Position', [650, y_pos, 750, y_pos+30]);

    % Connect from function block (branching)
    add_line(modelName, ['Boost_Converter/' num2str(i)], [outputs{i} '_display/1'], ...
        'autorouting', 'on');
end

%% Set model configuration
set_param(modelName, 'Solver', 'FixedStepDiscrete');
set_param(modelName, 'FixedStep', '1e-5');
set_param(modelName, 'StopTime', '1e-5');  % Single step evaluation

%% Initialize default parameter values
assignin('base', 'duty_cycle', 0.5);
assignin('base', 'v_in', 12);
assignin('base', 'load_r', 10);
assignin('base', 'f_sw', 50e3);
assignin('base', 'L', 100e-6);
assignin('base', 'C', 100e-6);

%% Save model
save_system(modelName);

fprintf('Simple boost converter model created successfully!\n');
fprintf('Model saved as: %s.slx\n', modelName);
fprintf('\nThis model uses averaged equations for fast evaluation.\n');
fprintf('\nDefault parameters:\n');
fprintf('  duty_cycle = 0.5\n');
fprintf('  v_in = 12 V\n');
fprintf('  load_r = 10 Ω\n');
fprintf('  f_sw = 50 kHz\n');
fprintf('  L = 100 µH\n');
fprintf('  C = 100 µF\n');
fprintf('\nOutputs:\n');
fprintf('  v_out = Output voltage (V)\n');
fprintf('  i_l = Inductor current (A)\n');
fprintf('  v_out_ripple = Voltage ripple (V)\n');
fprintf('  i_l_ripple = Current ripple (A)\n');

%% Test the model
fprintf('\nTesting model with default parameters...\n');
try
    simOut = sim(modelName);
    fprintf('✓ Model simulation successful!\n');
    fprintf('  V_out = %.2f V (expected ~%.2f V)\n', simOut.v_out, 12/(1-0.5));
    fprintf('  I_L = %.2f A\n', simOut.i_l);
    fprintf('  V_ripple = %.4f V\n', simOut.v_out_ripple);
    fprintf('  I_ripple = %.4f A\n', simOut.i_l_ripple);
catch ME
    fprintf('⚠ Could not test model automatically: %s\n', ME.message);
    fprintf('You can test it manually in Simulink.\n');
end

end
