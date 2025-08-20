% EXAMPLE SCRIPT FOR POST DLC TRACKING ANALYSIS
%
% Published as part of "Automated behavioral tracking of 
% zebrafish larvae with DeepLabCut and SLEAP: pre-trained
% networks and datasets of annotated poses"
%
% Leandro A. Scholz, Tessa Mancienne, Sarah J. Stednitz, Ethan K. Scott, Conrad C.Y. Lee
%
% Written by Conrad C.Y.Lee and Leandro A. Scholz
% University of Melbourne (2025)

clear vars
close all
clc

%% Locate the path where you had downloaded the functions
addpath(genpath(uigetdir('C:/', 'Select Function Path Directory')));

%% Select where your dlc output is located
dataPath = uigetdir;
files = dir(fullfile(dataPath, '*.h5'));
data = h5read(fullfile(dataPath, files.name),'/df_with_missing/table').values_block_0;

%% Defining parameters
% Table labels as defined by DeepLabCut
body_parts = ["swim_bladder" "tail_1" "tail_2" "tail_3" "tail_4" "tail_5" "tail_6" "tail_7" "tail_8" "tail_9" "tail_10" "R_eye_top" "R_eye_bottom" "L_eye_top" "L_eye_bottom"];
params = ["x" "y" "prob"];

% Choose what body parts you want to use to calculate the center and heading of the fish
body_center_parts = {'swim_bladder', 'R_eye_bottom','L_eye_bottom','R_eye_top','L_eye_top'};   % Choose swim_bladder only if tracking confidence of eyes are poor
heading_parts.origin = {'swim_bladder'};
heading_parts.point = {'R_eye_bottom','L_eye_bottom','R_eye_top','L_eye_top'};

% General Parameters 
thr = 0.9;                              % Threshold for rejecting poor tracking confidence
frame_rate = 300;                       % Frame rate: so all calculation is converted into seconds (e.g. pixel/s or mm/s)
smooth_val = 5;                         % Size of sliding average window. Values must be odd numbered
pxl_resolution=0.0800;                  % mm per pixel: so all calculations can be converted into mm (eg. mm/f or mm/s)
frame_resolution=[254 254];             % Size of video
tolerance = 0.015;                      % Tolerance used to remove noise
num_tail_seg = 9;                       % Number of tail segment used for MTC Calculations

% Parameters for bout detection
smooth_window = 5;                      % Size of smoothing window
min_peak_seperation = 0.1667;           % Minimium peak seperation in seconds
min_peak_prominence = 0.01;             % Minimium peak prominence (au)
bout_window_start = 60;                 % Size of window to look back to determine start of bout
bout_window_end = 100;                  % Size of window to look forward to determine end of bout (distance)
num_repeats = 7;                        % Number of frames of non-movement to determine fish is no longer moving
std_limit = 4;                          % Factor of standard deviation to define tail velocity noise

% Perform Calculations (Point-metrics, vector metrics, bout metrics - see Github Help for details)
[body_center, ~] = get_body_center(data,thr,body_parts,params,body_center_parts); 
point_metrics = get_point_metrics(body_center,tolerance=tolerance ,avg_confidence=thr,thr=thr,frame_rate=frame_rate,smooth_val=smooth_val,pxl_resolution=pxl_resolution,frame_resolution=frame_resolution);
vector_metrics = get_vector_metrics(data,body_parts,params,heading_parts,thr,num_tail_seg,frame_rate);
bout_metric = get_bout_metrics(point_metrics,vector_metrics,frame_rate,smooth_window,min_peak_seperation,min_peak_prominence,bout_window_start,bout_window_end,num_repeats,std_limit);

%% Basic Plotting
recording_duration = 6; % seconds
x = linspace(0,recording_duration,recording_duration*frame_rate);
x_v = x(1:end-1);

close all
figure
set(gcf,'Position', [200 200 1595 564])
subplot(2,4,[1 2 5 6])
plot(point_metrics.normalized_from_center(:,1),point_metrics.normalized_from_center(:,2),'k.')
xlim([-1 1])
ylim([-1 1])
axis square
title('XY Position over time')
set(gca,'TickDir','out')
ylabel('Normalized Y')
ylabel('Normalized X')

subplot(2,4,[3 4]); hold on
bar_height = 150;
for i = 1:length(bout_metric.start_locs)
    rectangle('Position',[x_v(bout_metric.start_locs(i)) 0 x_v(bout_metric.end_locs(i)-bout_metric.start_locs(i)) bar_height],'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.3, 'EdgeColor','None')
end
plot(x_v,point_metrics.velocity,'k')
ylabel('Speed (mm/s)')
xlabel('Time (s)')
ylim([0 bar_height])
title('Swim speed and bout locations')
set(gca,'TickDir','out','box','off')

subplot(2,4,[7 8]); hold on
bar_height = 100;
for i = 1:length(bout_metric.start_locs)
    rectangle('Position',[x(bout_metric.start_locs(i)) -bar_height x(bout_metric.end_locs(i)-bout_metric.start_locs(i)) bar_height*2],'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.3, 'EdgeColor','None')
end
plot(x,vector_metrics.mean_tail_angle,'k')
ylabel('Degrees')
xlabel('Time (s)')
ylim([-bar_height bar_height])
title('Mean tail angle and bout locations')
set(gca,'TickDir','out','box','off')
