function vector_metrics = get_vector_metrics(data,body_parts,params,heading_parts,thr,num_tail_seg,frame_rate)
%GET_VECTOR_METRICS Calculates a suite of kinematic metrics for animal locomotion.
%   VECTOR_METRICS = GET_VECTOR_METRICS(DATA, BODY_PARTS, PARAMS, HEADING_PARTS, THR, NUM_TAIL_SEG, FRAME_RATE)
%   computes various metrics, including heading, change in heading, tail angles, and tail
%   velocity/acceleration, from DeepLabCut tracking data. The function uses specified body
%   parts to define a reference vector for heading, and it validates tracking points
%   based on a confidence threshold.
%
%   Inputs:
%       - data:           A 2D matrix of raw tracking data, with columns representing frames
%                         and rows containing [x, y, prob] triplets for each body part.
%       - body_parts:     A cell array of strings listing all tracked body parts.
%       - params:         A cell array of strings defining the parameters for each body part
%                         (e.g., {'x', 'y', 'prob'}).
%       - heading_parts:  A structure with fields `origin` and `point` that specify the
%                         body parts used to calculate the animal's heading vector. `origin`
%                         is a cell array for the base of the vector, and `point` is a cell
%                         array for its tip.
%       - thr:            A scalar confidence threshold. Tracking points with a probability
%                         below this value are ignored.
%       - num_tail_seg:   The number of tail segments used to calculate tail curvature.
%       - frame_rate:     The frame rate of the video (in Hz), used for velocity and
%                         acceleration calculations.
%
%   Output:
%       - vector_metrics: A structure containing the calculated metrics:
%           - heading:           The instantaneous heading (orientation) in degrees (0 to 360).
%           - delta_heading:     The signed change in heading between consecutive frames,
%                                accounting for 360-degree wrap-around.
%           - abs_heading:       The absolute change in heading between consecutive frames.
%           - mean_tail_angle:   The average tail angle (e.g., a measure of tail curvature).
%           - mean_tail_vel:     The instantaneous velocity of the mean tail angle.
%           - mean_tail_acc:     The instantaneous acceleration of the mean tail angle.
%           - tail_seg_angles:   The angles of individual tail segments.
%           - tail_vectors_norm: The normalized vectors for each tail segment.
%
%   Example:
%       thr = 0.9;
%       num_tail_seg = 9;
%       frame_rate = 300;
%
%       % Define the body parts to be used for heading
%       heading_parts.origin = {'swim_bladder'};
%       heading_parts.point = {'R_eye_bottom','L_eye_bottom'};
%
%       % Compute the kinematic metrics
%       metrics = get_vector_metrics(data, body_parts, params, heading_parts, thr, num_tail_seg, frame_rate);
%
%   See also GET_BODY_CENTER, CAL_TAIL_ANGLE, CAL_DIFF_HEADING_DIR.

% Get points for calculation 
[origin_center, ~] = get_body_center(data,thr,body_parts,params,heading_parts.origin); 
[point_center, ~] = get_body_center(data,thr,body_parts,params,heading_parts.point); 
coord_diff = point_center'-origin_center';
coord_diff_norm = [coord_diff(:,1)./sqrt(coord_diff(:,1).^2+coord_diff(:,2).^2) coord_diff(:,2)./sqrt(coord_diff(:,1).^2+coord_diff(:,2).^2)];
ref = [1,0];

% Calculate Heading
dot_product = arrayfun(@(x) dot(coord_diff_norm(x,:),ref), 1:length(coord_diff_norm));
determinate = arrayfun(@(x) det([coord_diff_norm(x,:); ref]), 1:length(coord_diff_norm));
heading = rad2deg(atan2(determinate,dot_product)+pi);

% Calculate Change in Heading
delta_heading = arrayfun(@(x,y) cal_diff_heading_dir(x,y), heading(1:end-1), heading(2:end));
abs_delta_heading = min((360-abs(heading(1:end-1)-heading(2:end))),abs(heading(1:end-1)-heading(2:end)));

[tail_seg_angles, mean_tail_angle, tail_vectors_norm] = cal_tail_angle(data,body_parts,params,thr,coord_diff_norm,num_tail_seg);

mean_tail_vel = [0; diff(mean_tail_angle)/(1/frame_rate)];
mean_tail_acc = [0; diff(mean_tail_vel)/(1/frame_rate)];

% Output as structure
vector_metrics.heading = heading;
vector_metrics.delta_heading = delta_heading;
vector_metrics.abs_heading = abs_delta_heading;
vector_metrics.mean_tail_angle = mean_tail_angle;
vector_metrics.mean_tail_vel = mean_tail_vel;
vector_metrics.mean_tail_acc = mean_tail_acc;
vector_metrics.tail_seg_angles = tail_seg_angles;
vector_metrics.tail_vectors_norm = tail_vectors_norm;


