function hab = compute_habituation_metrics(animal, habSess)
% COMPUTE_HABITUATION_METRICS
% Runs a full habituation analysis battery for sessions 1:habSess
%
% INPUT
%   animal   : struct array (animal(a).session(s))
%   habSess  : number of habituation sessions (e.g., 16)
%
% OUTPUT
%   hab : struct containing all habituation metrics

if nargin < 2
    habSess = 16;
end

nAnim = numel(animal);
nSess = numel(habSess);

%% -------------------------
% Preallocate
% -------------------------
hab.meanSpeed        = nan(nAnim,nSess);
hab.deltaDay         = nan(nAnim,nSess-1);
hab.trialSlope       = nan(nAnim,nSess);
hab.immobileFrac     = nan(nAnim,nSess);
hab.earlyLateDiff    = nan(nAnim,nSess);
hab.ICC              = nan(1,nSess);

speedThresh = 1; % cm/s for immobility

%% -------------------------
% Main loop
% -------------------------
for a = 1:nAnim
    for s = habSess

        if s > numel(animal(a).session)
            continue
        end

        sess = animal(a).session(s);

        % ---------- Mean speed & trial slope ----------
        if isfield(sess,'T') && ~isempty(sess.T)
            hab.meanSpeed(a,s)  = mean(sess.T.AirOn_MeanSpeed,'omitnan');

            if isfield(sess,'perfMetrics')
                hab.trialSlope(a,s) = sess.perfMetrics.learningSlope;
            end

            % ---------- Early vs late ----------
            T = sess.T;
            nT = height(T);
            if nT >= 3
                early = 1:round(nT/3);
                late  = round(2*nT/3):nT;
                hab.earlyLateDiff(a,s) = ...
                    mean(T.AirOn_MeanSpeed(early),'omitnan') - ...
                    mean(T.AirOn_MeanSpeed(late),'omitnan');
            end
        end

        % ---------- Immobility ----------
        if isfield(sess,'b') && ~isempty(sess.b)
            hab.immobileFrac(a,s) = mean(sess.b.speed_path_cms < speedThresh);
        end
    end
end

%% -------------------------
% Day-to-day stability
% -------------------------
hab.deltaDay = abs(diff(hab.meanSpeed,1,2));

%% -------------------------
% ICC per session
% -------------------------
for s = 1:nSess
    y = hab.meanSpeed(:,s);
    if all(isnan(y))
        continue
    end

    tbl = table(y, categorical((1:nAnim)'), ...
        'VariableNames',{'Y','Animal'});

    try
        lme = fitlme(tbl,'Y ~ 1 + (1|Animal)');
        vc  = lme.CovarianceParameters;
        hab.ICC(s) = vc{1,1} / sum(vc{:,1});
    catch
        hab.ICC(s) = NaN;
    end
end

%% -------------------------
% Quick diagnostic plots (optional, comment out if not needed)
% -------------------------
figure('Name','Habituation diagnostics','Color','w');

subplot(3,2,1)
plot(hab.meanSpeed','o-'); title('Mean speed');
xlabel('Session'); ylabel('Speed');

subplot(3,2,2)
plot(hab.deltaDay','o-'); title('|Δ mean speed|');
xlabel('Session transition'); ylabel('|Δ|');

subplot(3,2,3)
plot(hab.trialSlope','o-'); title('Trial slope');
xlabel('Session'); ylabel('Slope');

subplot(3,2,4)
plot(hab.immobileFrac','o-'); title('Immobility fraction');
xlabel('Session'); ylabel('Fraction');

subplot(3,2,5)
plot(hab.earlyLateDiff','o-'); title('Early–late difference');
xlabel('Session'); ylabel('Δ speed');

subplot(3,2,6)
plot(hab.ICC,'o-'); title('ICC (trait emergence)');
xlabel('Session'); ylabel('ICC');

end
