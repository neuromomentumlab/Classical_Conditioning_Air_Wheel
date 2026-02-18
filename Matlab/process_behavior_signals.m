function animal = process_behavior_signals(animal)

if ~exist('animal','var')
    animal = evalin('base','animal')
end

n = 0;
cams = {'face','pupil','paws'};
%%
for a = 1:numel(animal)
    % b = load(animal(an).mat);
    % b = make_behavior_struct(matFile)
    fprintf('Processing %s\n', animal(a).ID);

    for s = 1:numel(animal(a).session)

        matFile = animal(a).session(s).mat;

        if isempty(matFile) || ~exist(matFile,'file')
            warning('%s %s: MAT file missing', ...
                animal(a).ID, animal(a).session(s).date);
            continue;
        end

        fprintf('Processing %s - Session %s\n', animal(a).ID,animal(a).session(s).date);

        % Build behavioral struct
        b = make_behavior_struct(matFile);

        % Store
        animal(a).session(s).b = b;

    end


    % for c = 1:numel(cams)
    %         cam = cams{c};
    % 
    %         % If we don't have this camera, skip
    %         if ~isfield(animal(an).video.h264, cam)
    %             continue;
    %         end
    % 
    %         h264_name = animal(an).video.h264.(cam);
    % 
    %         if isempty(h264_name)
    %             continue;
    %         end
    % 
    %         % Full input path
    %         in_file = h264_name;
    %         if ~exist(in_file,'file')
    %             warning('process_h264:FileNotFound', ...
    %                 'Could not find %s for animal %d', in_file, an);
    %             continue;
    %         end
    %         try
    %         thisfile = animal(an).video.led.(cam);
    %         b.led.(cam) = readtable(thisfile);
    %         b.led_sig.(cam) = extract_led_from_roi(b.led.(cam), 60);
    %         catch
    %             disp('LED signal not extracted')
    %         end

    % end
end




function b = make_behavior_struct(matFile)
% MAKE_BEHAVIOR_STRUCT
% Build behavioral struct 'b' from raw DAQ MAT file
%
% INPUT:
%   matFile : full path to recording_*.mat
%
% OUTPUT:
%   b : behavior struct

% -----------------------
% Load MAT file
% -----------------------
S = load(matFile);

X = S.X;
fs = S.fs;

% -----------------------
% Identify channels
% -----------------------
airIdx = find(strcmpi(S.channelNames,'Air'));
encIdx = find(strcmpi(S.channelNames,'EncCount'));

assert(~isempty(airIdx) && ~isempty(encIdx), ...
    'Air or EncCount channel missing');

air_raw = X(:,airIdx);
enc_raw = X(:,encIdx);

% -----------------------
% Basic timing
% -----------------------
b.fs = fs;
b.si = 1/fs;
b.number_of_samples = size(X,1);
b.t = S.t(:);
b.tm = b.t/60;

% -----------------------
% Air signal
% -----------------------
b.air_raw = air_raw(:);

% Threshold air (robust, data-driven)
thr = median(b.air_raw) + 3*mad(b.air_raw,1);
b.air_bin = b.air_raw > 0.5;

% Rising / falling edges
air_diff = diff([0; b.air_bin; 0]);
% Air_r = find(air_diff == 1);
% Air_f = find(air_diff == -1) - 1;
Air_f = find_falling_edge(b.air_raw,-0.5,2);
Air_r = find_rising_edge(b.air_raw,0.5,2);


[b.Air_r, b.Air_f, b.Air_edges_report] = sanitize_air_edges(Air_r, Air_f);

% -----------------------
% Encoder / kinematics
% -----------------------
b.encoderCount = enc_raw(:);
b.countsPerRev = S.countsPerRev;

% Distance (in revolutions)
b.dist_rev = double(b.encoderCount) / b.countsPerRev;

% -----------------------
% Encoder / kinematics (WINDOWED, BEHAVIORAL)
% -----------------------

wheelDiameter_cm = 32;
wheelCircumference_cm = pi * wheelDiameter_cm;
cmPerCount = wheelCircumference_cm / b.countsPerRev;

enc = double(b.encoderCount(:));
dEnc = [0; diff(enc)];          % signed per-sample encoder increments

% -----------------------
% Distance (always valid)
% -----------------------
b.dist_net_cm  = enc * cmPerCount;                 % signed displacement
b.dist_path_cm = cumsum(abs(dEnc)) * cmPerCount;  % total path length

% -----------------------
% Speed (define time scale!)
% -----------------------
win_ms = 100;                                     % <-- behavioral window
win = round(win_ms/1000 * b.fs);

% Windowed encoder displacement
dEnc_win = movsum(dEnc, win, 'omitnan');

% Speed = displacement / time
b.speed_net_cms  = (dEnc_win * cmPerCount) / (win / b.fs);
b.speed_path_cms = (abs(dEnc_win) * cmPerCount) / (win / b.fs);

b.speed_window_ms = win_ms;   % store metadata

% -----------------------
% Direction bias (unchanged logic, now stable)
% -----------------------
b.bias = (b.dist_net_cm(end) - b.dist_net_cm(1)) / ...
         (b.dist_path_cm(end) - b.dist_path_cm(1) + eps);


% -----------------------
% Metadata
% -----------------------
b.startTime = S.startTime;
b.device    = S.device;
b.channelNames = S.channelNames;

function [Air_r_clean, Air_f_clean, report] = sanitize_air_edges(Air_r, Air_f)
% SANITIZE_AIR_EDGES
% Ensures rising/falling air edges are aligned correctly.
%
% Rules:
% 1) First Air_r must be before first Air_f
% 2) Each Air_r has one Air_f after it
% 3) Lengths match at the end
%
% Returns:
%   Air_r_clean, Air_f_clean : cleaned edge vectors
%   report : struct describing what was fixed

    report = struct();
    report.initial_r = numel(Air_r);
    report.initial_f = numel(Air_f);
    report.actions   = {};

    Air_r = Air_r(:);
    Air_f = Air_f(:);

    % -------------------------
    % Rule 1: first rise < first fall
    % -------------------------
    if ~isempty(Air_r) && ~isempty(Air_f)
        if Air_r(1) > Air_f(1)
            Air_f(1) = [];
            report.actions{end+1} = 'Removed first Air_f (recording started during Air ON)';
        end
    end

    % -------------------------
    % Rule 2: enforce one-to-one pairing
    % -------------------------
    n = min(numel(Air_r), numel(Air_f));
    Air_r = Air_r(1:n);
    Air_f = Air_f(1:n);

    % Remove invalid pairs (Air_f <= Air_r)
    bad = Air_f <= Air_r;
    if any(bad)
        Air_r(bad) = [];
        Air_f(bad) = [];
        report.actions{end+1} = 'Removed non-positive duration trials';
    end

    % -------------------------
    % Final outputs
    % -------------------------
    Air_r_clean = Air_r;
    Air_f_clean = Air_f;

    report.final_r = numel(Air_r_clean);
    report.final_f = numel(Air_f_clean);

function dist = processEncodeSignals(chb,cha)
n = 0;
chat = cha > 2.5;
chbt = chb > 2.5;
encoderCount = 0;
valP = [chat(1) chbt(1)];
dist(1) = encoderCount;
for ii = 2:length(cha)
    valC = [chat(ii) chbt(ii)];
    if valC == valP
        valP = valC;
        dist(ii) = encoderCount;
        continue;
    end
    if isequal(valP,[0 0]) && isequal(valC,[1 0])
        encoderCount = encoderCount + 1;
        valP = valC;
        dist(ii) = encoderCount;
        continue;
    end
    if isequal(valP,[0 0]) && isequal(valC,[0 1])
        encoderCount = encoderCount - 1;
        valP = valC;
        dist(ii) = encoderCount;
        continue;
    end
    if isequal(valP,[1 0]) && isequal(valC,[1 1])
        encoderCount = encoderCount + 1;
        valP = valC;
        dist(ii) = encoderCount;
        continue;
    end
    if isequal(valP,[1 0]) && isequal(valC,[0 0])
        encoderCount = encoderCount - 1;
        valP = valC;
        dist(ii) = encoderCount;
        continue;
    end
    if isequal(valP,[0 1]) && isequal(valC,[0 0])
        encoderCount = encoderCount + 1;
        valP = valC;
        dist(ii) = encoderCount;
        continue;
    end
    if isequal(valP,[0 1]) && isequal(valC,[1 1])
        encoderCount = encoderCount - 1;
        valP = valC;
        dist(ii) = encoderCount;
        continue;
    end
    if isequal(valP,[1 1]) && isequal(valC,[0 1])
        encoderCount = encoderCount + 1;
        valP = valC;
        dist(ii) = encoderCount;
        continue;
    end
    if isequal(valP,[1 1]) && isequal(valC,[1 0])
        encoderCount = encoderCount - 1;
        valP = valC;
        dist(ii) = encoderCount;
        continue;
    end
end
dist = dist';

