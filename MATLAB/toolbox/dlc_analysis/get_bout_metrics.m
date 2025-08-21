function bout_metric = get_bout_metrics(point_metrics,vector_metrics,frame_rate,smooth_window,min_peak_seperation,min_peak_prominence,bout_window_start,bout_window_end,num_repeats,std_limit)
%GET_BOUT_METRICS Identifies and analyzes individual movement bouts.
%   BOUT_METRIC = GET_BOUT_METRICS(POINT_METRICS, VECTOR_METRICS, FRAME_RATE, ...)
%   detects and characterizes individual "bouts" of movement (e.g., swimming or
%   locomotion) based on smoothed distance and kinematic data. It uses a peak-finding
%   algorithm to locate the peak of each bout and then defines the start and end of the
%   bout based on periods of low or no movement.
%
%   Inputs:
%       - point_metrics:        A structure containing point-based kinematic data,
%                               including `.distance_mm`, `.velocity`, and `.acceleration`.
%       - vector_metrics:       A structure containing vector-based metrics, including
%                               `.mean_tail_vel`, `.mean_tail_angle`, and `.delta_heading`.
%       - frame_rate:           The video frame rate in Hz.
%       - smooth_window:        The size of the smoothing window (odd integer) for
%                               `quick_smooth`. Defaults to 5.
%       - min_peak_seperation:  Minimum time in seconds between movement peaks.
%                               Defaults to 0.1667.
%       - min_peak_prominence:  Minimum prominence of a peak (in arbitrary units) to be
%                               considered a valid bout. Defaults to 0.01.
%       - bout_window_start:    Number of frames to look back from a peak to find the
%                               start of a bout. Defaults to 60.
%       - bout_window_end:      Number of frames to look forward from a peak to find the
%                               end of a bout. Defaults to 100.
%       - num_repeats:          Number of consecutive frames of low/zero movement
%                               required to define the start or end of a bout.
%                               Defaults to 7.
%       - std_limit:            A factor of the standard deviation used to define the
%                               threshold for tail velocity noise. Defaults to 4.
%
%   Output:
%       - bout_metric: A structure containing various metrics for each detected bout:
%           - start_locs:           Frame index of the bout's start.
%           - end_locs:             Frame index of the bout's end (based on distance).
%           - tail_end_locs:        Frame index of the bout's end (based on tail movement).
%           - total_distance:       Total distance traveled during the bout.
%           - mean_distance:        Average distance traveled per frame during the bout.
%           - duration_distance:    Duration of the bout in seconds (based on distance).
%           - duration_tail:        Duration of the bout in seconds (based on tail movement).
%           - avg_speed:            Average speed during the bout.
%           - max_speed:            Maximum speed during the bout.
%           - max_acceleration:     Maximum acceleration during the bout.
%           - tail_vel:             Average absolute tail velocity during the bout.
%           - tail_ang:             Average tail angle during the bout.
%           - raw_tail_vel:         A cell array of the raw tail velocity trace for each bout.
%           - raw_tail_ang:         A cell array of the raw tail angle trace for each bout.
%           - delta_heading:        A cell array of the raw change in heading for each bout.
%
%   Example:
%       % Assume `point_metrics` and `vector_metrics` structures are populated
%       % and `frame_rate` is 300.
%       % The function will use the default parameters for bout detection.
%       bout_data = get_bout_metrics(point_metrics, vector_metrics, 300, [], [], [], [], [], [], []);
%
%   See also QUICK_SMOOTH, FIND_REPEATS.

if isempty(smooth_window)
    smooth_window = 5;
end

if isempty(min_peak_seperation)
    min_peak_seperation = 0.1667;
end

if isempty(min_peak_prominence)
    min_peak_prominence = 0.01;
end

if isempty(bout_window_start)
    bout_window_start = 60;
end

if isempty(bout_window_end)
    bout_window_end = 100;
end

if isempty(num_repeats)
    num_repeats = 7;
end

if isempty(std_limit)
    std_limit = 4;
end

%% Grab vector metrics
distance_mm = point_metrics.distance_mm;
velocity = point_metrics.velocity;
mean_tail_vel = vector_metrics.mean_tail_vel;
mean_tail_angle = vector_metrics.mean_tail_angle;
acceleration = point_metrics.acceleration;
delta_heading = vector_metrics.delta_heading;

%% 1) Locate the bouts
distance_mm_smoothed = quick_smooth(distance_mm,smooth_window);
[~,locs] = findpeaks(distance_mm_smoothed,'MinPeakDistance',min_peak_seperation*frame_rate,'MinPeakProminence',min_peak_prominence);

% Remove bouts that are too close to the start and end
trimmed_locs = locs;
bad_bouts = [];
for i = 1:length(locs)
    if locs(i)-bout_window_start<=0
        bad_bouts = [bad_bouts i];
    end
    if locs(i)+bout_window_end>=length(distance_mm_smoothed)
        bad_bouts = [bad_bouts i];
    end
end
bad_bouts = unique(bad_bouts);
trimmed_locs(bad_bouts) = NaN;
trimmed_locs = trimmed_locs(~isnan(trimmed_locs));

% Remove bouts where there is bad tracking (50% of before are nans or 50% after are nans)
bad_bouts = [];
for i = 1:length(trimmed_locs)
    window_of_interest = distance_mm_smoothed(trimmed_locs(i)-bout_window_start:(trimmed_locs(i)));
    if sum(isnan(window_of_interest))/length(window_of_interest)>=0.5
        bad_bouts = [bad_bouts i];
    end
    window_of_interest = distance_mm_smoothed(trimmed_locs(i):trimmed_locs(i)+bout_window_end);
    if sum(isnan(window_of_interest))/length(window_of_interest)>=0.5
        bad_bouts = [bad_bouts i];
    end
end
bad_bouts = unique(bad_bouts);
trimmed_locs(bad_bouts) = NaN;
trimmed_locs = trimmed_locs(~isnan(trimmed_locs));


% 2) Determine where the bout begins
bout_start_locs = nan(1,length(trimmed_locs));
for i = 1:length(trimmed_locs)
    window_of_interest = distance_mm_smoothed(trimmed_locs(i)-bout_window_start:(trimmed_locs(i)));
    window_of_interest(isnan(window_of_interest))= 99;
    repeat_array = find_repeats(window_of_interest);
    start_temp = (trimmed_locs(i)-bout_window_start)+find((repeat_array>=num_repeats)+(window_of_interest==0)==2,1,'last')-1;
    if isempty(start_temp)
        [~,max_i] = max(diff(diff(window_of_interest)));
        bout_start_locs(i) = ((trimmed_locs(i)-bout_window_start)+max_i-1)+(smooth_window-1);
    else
        bout_start_locs(i) = start_temp+(smooth_window-1);
    end
end

% Check for over bout detection (bout begins at or after the peak of bout)
bad_bouts = [];
for i = 1:length(trimmed_locs)
    if trimmed_locs(i)-bout_start_locs(i)<=0
        bad_bouts = [bad_bouts i];
    end
end
bad_bouts = unique(bad_bouts);
trimmed_locs(bad_bouts) = NaN;
trimmed_locs = trimmed_locs(~isnan(trimmed_locs));
bout_start_locs(bad_bouts) = NaN;
bout_start_locs = bout_start_locs(~isnan(bout_start_locs));


% 3) Determine where the bout (distance) ends
bout_end_locs = nan(1,length(trimmed_locs));
for i = 1:length(trimmed_locs)
    window_of_interest = distance_mm_smoothed(trimmed_locs(i):trimmed_locs(i)+bout_window_end);
    window_of_interest(isnan(window_of_interest))= 99;
    repeat_array = find_repeats(window_of_interest);
    end_temp = trimmed_locs(i)+find((repeat_array>=num_repeats)+(window_of_interest==0)==2,1,'first')-1;
    if isempty(end_temp)
        if i == length(trimmed_locs)
            bout_end_locs(i) = length(distance_mm_smoothed);
        else
            bout_end_locs(i) = bout_start_locs(i+1);
        end
    else
        bout_end_locs(i) = end_temp;
    end
end

% 4) Determine where the bout (tail) ends
tail_end_locs = nan(1,length(trimmed_locs));
for bout_num = 1:length(trimmed_locs)
    original = mean_tail_vel(bout_start_locs(bout_num):bout_end_locs(bout_num));
    if sum(isnan(original)) == length(original)
        tail_end_locs(bout_num) = bout_end_locs(bout_num);
    else
        smoothed = quick_smooth(mean_tail_vel,smooth_window);
        smoothed = smoothed(bout_start_locs(bout_num):bout_end_locs(bout_num));

        smoothed_cut = smoothed;
        smoothed_cut((smoothed_cut<(std(smoothed)/std_limit) & smoothed_cut>-(std(smoothed)/std_limit))) = 0;
        cut_matrix = [smoothed_cut [smoothed_cut(2:end); NaN] [NaN; smoothed_cut(1:end-1)]];
        cut_matrix((sum(cut_matrix(:,2:3),2,'omitnan')==0),1)=0;
        cut_trace = cut_matrix(:,1);

        start_offset = trimmed_locs(bout_num)-bout_start_locs(bout_num)+1;
        focused_trace = cut_trace(start_offset:end)';
        repeat_array = find_repeats(focused_trace);
        end_temp = (find((repeat_array>=num_repeats)+(focused_trace==0)==2,1,'first')-1)+start_offset+smooth_window;
        if end_temp>length(original)
            tail_end_locs(bout_num) = length(original)+bout_start_locs(bout_num);
        elseif isempty(end_temp)
            tail_end_locs(bout_num) = length(original)+bout_start_locs(bout_num);
        else
            tail_end_locs(bout_num) = end_temp+bout_start_locs(bout_num);
        end
    end
end

% 5) Calculate the distance travelled for each bout
bout_total_distance = arrayfun(@(x) sum(distance_mm(bout_start_locs(x):bout_end_locs(x)),'omitnan'), 1:length(trimmed_locs));
bout_mean_distance = arrayfun(@(x) mean(distance_mm(bout_start_locs(x):bout_end_locs(x)),'omitnan'), 1:length(trimmed_locs));
bout_duration_distance = arrayfun(@(x) (bout_end_locs(x)-bout_start_locs(x))/frame_rate, 1:length(trimmed_locs));

% 6) Calculate the bout duration
bout_duration_tail = arrayfun(@(x) (tail_end_locs(x)-bout_start_locs(x))/frame_rate, 1:length(trimmed_locs));

% 7) Calculate average and max speed
bout_avg_speed = arrayfun(@(x) mean(velocity(bout_start_locs(x):bout_end_locs(x)),'omitnan'), 1:length(trimmed_locs));
% bout_max_speed = arrayfun(@(x) max(velocity(bout_start_locs(x):bout_end_locs(x))), 1:length(trimmed_locs));
bout_max_speed = cell2mat(arrayfun(@(x) max(velocity(bout_start_locs(x):bout_end_locs(x))), 1:length(trimmed_locs), 'UniformOutput', false));

% 8) Calculate avg tail velocity of each bout
bout_tail_vel = arrayfun(@(x) mean(abs(mean_tail_vel(bout_start_locs(x):tail_end_locs(x))),'omitnan'), 1:length(trimmed_locs));
bout_tail_ang = arrayfun(@(x) mean(mean_tail_angle(bout_start_locs(x):tail_end_locs(x)),'omitnan'), 1:length(trimmed_locs));

% 9) Output raw mean_tail velocity of each bout
raw_tail_vel = arrayfun(@(x) mean_tail_vel(bout_start_locs(x):tail_end_locs(x)), 1:length(trimmed_locs), 'UniformOutput', false);

% 10) Output raw mean_tail velocity of each bout
raw_tail_ang = arrayfun(@(x) mean_tail_angle(bout_start_locs(x):tail_end_locs(x)), 1:length(trimmed_locs), 'UniformOutput', false);

% 11) Output max acceleration
bout_max_acceleration = cell2mat(arrayfun(@(x) max(acceleration(bout_start_locs(x):bout_end_locs(x))), 1:length(trimmed_locs), 'UniformOutput', false));


% 12) 
bout_delta_heading = arrayfun(@(x) delta_heading(bout_start_locs(x):tail_end_locs(x)), 1:length(trimmed_locs), 'UniformOutput', false);


% Export as structure
bout_metric.start_locs = bout_start_locs;
bout_metric.end_locs = bout_end_locs;
bout_metric.tail_end_locs = tail_end_locs;
bout_metric.total_distance = bout_total_distance;
bout_metric.mean_distance = bout_mean_distance;
bout_metric.duration_distance = bout_duration_distance;
bout_metric.duration_tail = bout_duration_tail;
bout_metric.avg_speed = bout_avg_speed;
bout_metric.max_speed = bout_max_speed;
bout_metric.max_acceleration = bout_max_acceleration;
bout_metric.tail_vel = bout_tail_vel;
bout_metric.tail_ang = bout_tail_ang;
bout_metric.raw_tail_vel = raw_tail_vel;
bout_metric.raw_tail_ang = raw_tail_ang;
bout_metric.raw_tail_ang = raw_tail_ang;
bout_metric.delta_heading = bout_delta_heading;

