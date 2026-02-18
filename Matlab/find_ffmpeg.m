function ffmpeg = find_ffmpeg()

ffmpeg = '';

ispc = evalin("base",'mData.ispc');

if ispc
    [status, out] = system('where ffmpeg');
else
    [status, out] = system('which ffmpeg');
end

if status == 0
    ffmpeg = strtrim(out);
    ffmpeg = strtok(ffmpeg, newline); % take first match
end
end
