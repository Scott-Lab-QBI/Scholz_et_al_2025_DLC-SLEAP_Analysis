function [tail_seg_angles, mean_tail_angle, tail_vectors_norm] = cal_tail_angle(data,body_parts,params,thr,coord_diff_norm,num_tail_seg)
%CAL_TAIL_ANGLE Calculates the angle of individual tail segments and mean tail curvature.
%   [TAIL_SEG_ANGLES, MEAN_TAIL_ANGLE, TAIL_VECTORS_NORM] = CAL_TAIL_ANGLE(...)
%   computes the angles of each tail segment relative to the animal's main body axis.
%   This function is typically used to quantify tail bending during locomotion or
%   turning maneuvers. The angles are calculated by first defining vectors for each
%   tail segment and then comparing them to a reference vector (the `coord_diff_norm`).
%
%   Inputs:
%       - data:               A 2D matrix of raw tracking data from DeepLabCut.
%       - body_parts:         A cell array of strings listing all tracked body parts.
%       - params:             A cell array of strings defining the parameters for each
%                             body part (e.g., {'x', 'y', 'prob'}).
%       - thr:                A scalar confidence threshold. Tracking points with a probability
%                             below this value are ignored and treated as NaN.
%       - coord_diff_norm:    A normalized 2D vector representing the animal's body axis.
%                             This is typically computed from a main body part (e.g., swim bladder)
%                             to another point (e.g., an average of eye positions).
%       - num_tail_seg:       The number of tail segments to include in the calculation.
%
%   Outputs:
%       - tail_seg_angles:    An N x `num_tail_seg` matrix where N is the number of frames.
%                             Each column contains the angle of one tail segment in degrees.
%       - mean_tail_angle:    An N x 1 vector containing the average tail angle across all
%                             segments for each frame. This is a robust measure of overall
%                             tail curvature.
%       - tail_vectors_norm:  A structure with fields `x` and `y` containing the normalized
%                             x and y components of the vectors for each tail segment.
%
%   Example:
%       % Assume `data`, `body_parts`, `params` are defined and `coord_diff_norm` is
%       % a normalized vector representing the body axis.
%       thr = 0.9;
%       num_tail_seg = 9;
%
%       % Calculate the tail metrics
%       [seg_angles, mean_angle, vectors] = cal_tail_angle(data, body_parts, params, thr, coord_diff_norm, num_tail_seg);
%
%       % The `mean_angle` variable now contains the average tail angle for each frame.
%
%   See also GET_BODY_CENTER.

% Get the tail points
tail_selection = body_parts(1:num_tail_seg+1);
tail_points_xy = cellfun(@(x) get_body_center(data,thr,body_parts,params,{x}), tail_selection,'UniformOutput',false);
tail_points_mat = cell2mat(tail_points_xy')';

% Calculate the vectors
tail_vectors_x = tail_points_mat(:,3:2:end)-tail_points_mat(:,1:2:end-2);
tail_vectors_y = tail_points_mat(:,4:2:end)-tail_points_mat(:,2:2:end-2);
tail_vectors_norm.x = tail_vectors_x./sqrt(tail_vectors_x.^2 + tail_vectors_y.^2);
tail_vectors_norm.y = tail_vectors_y./sqrt(tail_vectors_x.^2 + tail_vectors_y.^2);

tail_norm_mat = [reshape(tail_vectors_norm.x,[],1) reshape(tail_vectors_norm.y,[],1)];
tail_norm_mat = [tail_norm_mat zeros(size(tail_norm_mat,1),1)];

% Prepare the Heading
heading_norm_mat = repmat(-coord_diff_norm,num_tail_seg,1);
heading_norm_mat = [heading_norm_mat zeros(size(heading_norm_mat,1),1)];

% Calculate the angles
dot_product = dot(tail_norm_mat,heading_norm_mat,2);
cross_product = cross(tail_norm_mat,heading_norm_mat);
dot_cross_product = dot(cross_product,repmat([0 0 1],size(cross_product,1),1),2);
all_tail_angles = rad2deg(atan2(dot_cross_product,dot_product));

tail_seg_angles = reshape(all_tail_angles,[],num_tail_seg);
mean_tail_angle = mean(tail_seg_angles,2);


