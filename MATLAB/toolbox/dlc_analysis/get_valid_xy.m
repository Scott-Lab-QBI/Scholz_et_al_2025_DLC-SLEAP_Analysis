function valid_points = get_valid_xy(data,body_parts,params,part,thr)
%GET_VALID_XY Extracts and validates x-y coordinates for a specific body part.
%   VALID_POINTS = GET_VALID_XY(DATA, PART, THR) takes raw tracking data from
%   DeepLabCut, filters out coordinates with a confidence score below a
%   specified threshold, and returns a 2xN matrix of valid x and y
%   coordinates. Invalid points are replaced with NaN to handle missing data
%   gracefully in subsequent calculations.
%
%   Inputs:
%       - data: A 2D matrix where each column represents a frame. The rows
%         are organized as [x, y, prob] triplets for each body part.
%         e.g., [x_part1, y_part1, prob_part1, x_part2, y_part2, ...].
%       - part: A string specifying the name of the body part (e.g., "tail_1").
%       - thr: A scalar value representing the confidence threshold. Points with a
%         probability below this value will be marked as invalid.
%
%   Output:
%       - valid_points: A 2xN matrix where the first row contains the valid x-coordinates
%         and the second row contains the valid y-coordinates. Invalid coordinates are
%         replaced with NaN.
%
%   Example:
%       % Assume `data` is a 2D matrix of DLC output
%       % and `get_parts_index` is available in the path.
%       % Get the valid (x,y) coordinates for the 'swim_bladder' with a threshold of 0.9.
%       valid_bladder_coords = get_valid_xy(data, "swim_bladder", 0.9);
%
%   See also GET_PARTS_INDEX.

invalid_frames = data(get_parts_index(body_parts,params,part,'prob'),:)<thr;
x_pos = data(get_parts_index(body_parts,params,part,'x'),:);
x_pos(invalid_frames) = NaN;
y_pos = data(get_parts_index(body_parts,params,part,'y'),:);
y_pos(invalid_frames) = NaN;
valid_points = [x_pos; y_pos];
