function build_hifi_switching_model()
% BUILD_HIFI_SWITCHING_MODEL Open the included Simscape switching model.
%
% The public repository already includes the configured model file
% boost_switching.slx. This helper loads the model and assigns the default
% workspace variables used by the validation script.

this_dir = fileparts(mfilename('fullpath'));
model_file = fullfile(this_dir, 'boost_switching.slx');
model_name = 'boost_switching';

if ~exist(model_file, 'file')
    error('Model file not found: %s', model_file);
end

assignin('base', 'duty_cycle', 0.5);
assignin('base', 'v_in', 12);
assignin('base', 'load_r', 10);
assignin('base', 'f_sw', 50e3);
assignin('base', 'L', 100e-6);
assignin('base', 'C', 100e-6);

load_system(model_file);
open_system(model_name);

fprintf('Opened %s with default workspace variables loaded.\n', model_name);
fprintf('Model file: %s\n', model_file);
end
