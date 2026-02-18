function [perf, T, perfMetrics] = compute_air_trial_performance(b)
% COMPUTE_AIR_TRIAL_PERFORMANCE
% Computes per-trial behavioral performance metrics during
% Air ON and Air OFF phases using PATH LENGTH kinematics.
%
% INPUT:
%   b : behavior struct containing
%       - Air_r, Air_f
%       - speed_path_cms
%       - dist_path_cm
%       - fs
%
% OUTPUT:
%   perf        : per-trial performance struct
%   T           : per-trial table
%   perfMetrics : session-level summary metrics

% ------------------------
% Default outputs (safe)
% ------------------------
perf = struct();
T = table();
perfMetrics = struct();

% ------------------------
% Basic validation
% ------------------------
if isempty(b) || ...
   ~isfield(b,'Air_r') || ~isfield(b,'Air_f') || ...
   ~isfield(b,'speed_path_cms') || ...
   ~isfield(b,'dist_path_cm') || ...
   ~isfield(b,'fs')

    warning('compute_air_trial_performance: invalid or empty b');
    return;
end

% ------------------------
% Sanitize Air edges
% ------------------------
Air_r = b.Air_r(:);
Air_f = b.Air_f(:);

% Ensure first rise < first fall
if ~isempty(Air_r) && ~isempty(Air_f) && Air_r(1) > Air_f(1)
    Air_f(1) = [];
end

n = min(numel(Air_r), numel(Air_f));
Air_r = Air_r(1:n);
Air_f = Air_f(1:n);

valid = Air_f > Air_r;
Air_r = Air_r(valid);
Air_f = Air_f(valid);

nTrials = numel(Air_r);
if nTrials == 0
    warning('No valid Air trials found');
    return;
end

% ------------------------
% Preallocate
% ------------------------
perf.nTrials = nTrials;

perf.airOn.meanSpeed  = nan(nTrials,1);
perf.airOn.distance   = nan(nTrials,1);
perf.airOn.duration   = nan(nTrials,1);

perf.airOff.meanSpeed = nan(nTrials,1);
perf.airOff.distance  = nan(nTrials,1);
perf.airOff.duration  = nan(nTrials,1);

% ------------------------
% Trial loop
% ------------------------
for i = 1:nTrials

    % ===== AIR ON =====
    idx_on = Air_r(i):Air_f(i);

    perf.airOn.meanSpeed(i) = mean(b.speed_path_cms(idx_on), 'omitnan');
    perf.airOn.distance(i)  = ...
        b.dist_path_cm(idx_on(end)) - b.dist_path_cm(idx_on(1));
    perf.airOn.duration(i)  = numel(idx_on) / b.fs;

    % ===== AIR OFF =====
    if i == 1
        idx_off = 1:(Air_r(i)-1);
    else
        idx_off = Air_f(i-1):(Air_r(i)-1);
    end

    if ~isempty(idx_off)
        perf.airOff.meanSpeed(i) = mean(b.speed_path_cms(idx_off), 'omitnan');
        perf.airOff.distance(i)  = ...
            b.dist_path_cm(idx_off(end)) - b.dist_path_cm(idx_off(1));
        perf.airOff.duration(i)  = numel(idx_off) / b.fs;
    end
end

% ------------------------
% Assemble table
% ------------------------
perf.trial = (1:nTrials)';
T = air_performance_table(perf);

% ------------------------
% Session-level metrics
% ------------------------
perfMetrics.meanARI = mean(T.ARI_Speed, 'omitnan');
perfMetrics.meanDGR = mean(T.DGR, 'omitnan');

perfMetrics.meanEfficiency_ON = mean(T.Efficiency_ON, 'omitnan');
perfMetrics.cvEfficiency_ON = ...
    std(T.Efficiency_ON, 'omitnan') / ...
    mean(T.Efficiency_ON, 'omitnan');

% Learning / fatigue slope
p = polyfit(T.Trial, T.AirOn_MeanSpeed, 1);
perfMetrics.learningSlope = p(1);

% Early vs late
n = height(T);
early = 1:round(n/3);
late  = round(2*n/3):n;

perfMetrics.earlyMeanSpeed = mean(T.AirOn_MeanSpeed(early), 'omitnan');
perfMetrics.lateMeanSpeed  = mean(T.AirOn_MeanSpeed(late),  'omitnan');
perfMetrics.deltaEarlyLate = ...
    perfMetrics.lateMeanSpeed - perfMetrics.earlyMeanSpeed;

end



function T = air_performance_table(perf)

T = table();
T.Trial = perf.trial;

% -------- Air ON --------
T.AirOn_MeanSpeed = perf.airOn.meanSpeed;
T.AirOn_Distance  = perf.airOn.distance;
T.AirOn_Duration  = perf.airOn.duration;

% -------- Air OFF --------
T.AirOff_MeanSpeed = perf.airOff.meanSpeed;
T.AirOff_Distance  = perf.airOff.distance;
T.AirOff_Duration  = perf.airOff.duration;

% -------- Differences --------
T.Delta_MeanSpeed = T.AirOn_MeanSpeed - T.AirOff_MeanSpeed;
T.Delta_Distance  = T.AirOn_Distance  - T.AirOff_Distance;

% -------- Indices --------
T.ARI_Speed = ...
    (T.AirOn_MeanSpeed - T.AirOff_MeanSpeed) ./ ...
    (T.AirOn_MeanSpeed + T.AirOff_MeanSpeed + eps);

T.DGR = ...
    (T.AirOn_Distance ./ T.AirOn_Duration) ./ ...
    (T.AirOff_Distance ./ T.AirOff_Duration + eps);

T.Efficiency_ON = T.AirOn_Distance ./ T.AirOn_Duration;

end
