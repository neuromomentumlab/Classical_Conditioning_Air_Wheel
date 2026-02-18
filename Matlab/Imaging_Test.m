%%
clear all
%%
main_dir = 'E:\GoogleDrive\InayatSamsoon\UNLV\data_2P\Texas_Red_Dextran_2026_01_15';
pdata_dir = fullfile(main_dir,'PData'); mD.pdata_dir = pdata_dir;
if ~exist(pdata_dir,"dir")
    mkdir(pdata_dir);
end
%%

rdata_dir = 'X:\Research\Neuromomentum_Cognemotion Lab\Raw_Data_Backup\data_2P\Texas_Red_Dextran_2026_01_15';
rdata_dir = 'E:\GoogleDrive\InayatSamsoon\UNLV\data_2P\Texas_Red_Dextran_2026_01_15\RData\Texas_Red_Dextran_2026_01_15';

filename = fullfile(rdata_dir,'file_00003.tif');

gifFile = fullfile(pdata_dir,'meanZ_stack_3.gif');

info = imfinfo(filename);
nFrames = numel(info);

fprintf('Total frames in TIFF: %d\n', nFrames);

t = Tiff(filename, 'r');

% Read first frame to get image size
img0 = imread(filename, 1);
[ny, nx] = size(img0);

% Preallocate
stack = zeros(ny, nx, nFrames, 'like', img0);

% Load all frames
for k = 1:nFrames
    stack(:,:,k) = imread(filename, k);
end

%
T = 10;                       % frames per slice
Z = nFrames / T;              % number of slices

assert(mod(nFrames, T) == 0, 'Frame count not divisible by T');

% Reshape: [Y X Z T]
stackZT = reshape(stack, ny, nx, T, Z);

% Permute to [Y X Z T]
stackZT = permute(stackZT, [1 2 4 3]);
%
z = 50;   % slice index
t = 1;    % frame index

figure(100);clf
imagesc(stackZT(:,:,z,t));
axis image off;
colormap gray;
title(sprintf('Z = %d, T = %d', z, t));

%
meanZ = mean(stackZT, 4);

figure(100);clf
for z = 1:Z
    imagesc(meanZ(:,:,z));
    axis image off;
    colormap gray;
    title(sprintf('Z slice %d (%.1f µm)', z, (z-1)));
    drawnow;
end

%
delayTime = 0.15;   % seconds between frames
clim = prctile(meanZ(:), [1 99]);
figure(100);clf
colormap gray;

for z = 1:Z
    imagesc(meanZ(:,:,z), clim);
    axis image off;
    title(sprintf('Z slice %d (%.1f µm)', z, (z-1)));
    drawnow;

    % Capture frame
    frame = getframe(gcf);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);

    % Write to GIF
    if z == 1
        imwrite(imind, cm, gifFile, 'gif', ...
                'Loopcount', inf, 'DelayTime', delayTime);
    else
        imwrite(imind, cm, gifFile, 'gif', ...
                'WriteMode', 'append', 'DelayTime', delayTime);
    end
end
