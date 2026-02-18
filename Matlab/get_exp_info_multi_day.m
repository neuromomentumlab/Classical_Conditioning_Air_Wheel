function animal = get_exp_info_multi_day(mD, animal_list)

rdata_dir = mD.rdata_dir;
pdata_dir = mD.pdata_dir;
adata_dir = mD.adata_dir;



sessionTemplate = struct( ...
    'date','', ...
    'rdir','', ...
    'pdir','', ...
    'adir','', ...
    'video', struct( ...
        'h264', struct( ...
            'face','', ...
            'pupil','', ...
            'paws','' ) ), ...
    'mat','' );

for a = 1:numel(animal_list)

    animal(a).ID = animal_list{a};
    animal(a).session = sessionTemplate([]);

    animal_dir = fullfile(rdata_dir, animal(a).ID);

    % if ~exist(animal_dir,'dir')
    %     warning('Animal folder not found: %s', animal_dir);
    %     continue;
    % end

    % --------------------------------------------------
    % Discover date folders automatically
    % --------------------------------------------------
    d = dir(animal_dir);
    isDateDir = [d.isdir] & ...
                ~ismember({d.name},{'.','..'});

    dateDirs = {d(isDateDir).name};

    % Optional: enforce YYYY_MM_DD pattern
    dateDirs = dateDirs(~cellfun(@isempty, ...
        regexp(dateDirs,'^\d{4}_\d{2}_\d{2}$')));

    % Sort chronologically
    dateNums = datenum(dateDirs,'yyyy_mm_dd');
    [~,idx] = sort(dateNums);
    dateDirs = dateDirs(idx);
    
    % --------------------------------------------------
    % Loop over sessions
    % --------------------------------------------------
    for s = 1:numel(dateDirs)
        sess = sessionTemplate;   % reset structure
        tdate = dateDirs{s};

        sess.date = tdate;
        sess.rdir = fullfile(rdata_dir, animal(a).ID, tdate);
        sess.pdir = fullfile(pdata_dir, animal(a).ID, tdate);
        if ~exist(sess.pdir,'dir')
            mkdir(sess.pdir)
        end
        sess.adir = fullfile(adata_dir, animal(a).ID, tdate);
        if ~exist(sess.adir,'dir')
            mkdir(sess.adir)
        end
        % if ~exist(sess.pdir,'dir'), mkdir(sess.pdir); end
        % if ~exist(sess.adir,'dir'), mkdir(sess.adir); end

        files = dir(fullfile(sess.rdir,'*'));

        % Initialize
        sess.video.h264.face  = '';
        sess.video.h264.pupil = '';
        sess.video.h264.paws  = '';
        sess.mat = '';

        for fn = 1:numel(files)

            fname = files(fn).name;
            fullfname = fullfile(sess.rdir,fname);

            if files(fn).isdir
                continue;
            end

            % ---- H264 ----
            if endsWith(fname,'.h264','IgnoreCase',true)
                if startsWith(fname,'face','IgnoreCase',true)
                    sess.video.h264.face = fullfname;
                elseif startsWith(fname,'pupi','IgnoreCase',true)
                    sess.video.h264.pupil = fullfname;
                elseif startsWith(fname,'video','IgnoreCase',true)
                    sess.video.h264.paws = fullfname;
                end

            % ---- MAT ----
            elseif endsWith(fname,'.mat','IgnoreCase',true)
                sess.mat = fullfname;
            end
        end

        animal(a).session(end+1) = sess;
    end
end
end
