function [body_center, avg_confidence] = get_body_center(data,thr,body_parts,params,body_elements)
%GET_BODY_CENTER Calculates the geometric center and average confidence of a set of body parts.
%   [BODY_CENTER, AVG_CONFIDENCE] = GET_BODY_CENTER(DATA, THR, BODY_ELEMENTS)
%   computes the average x and y position (the geometric center) across a
%   specified group of body parts for each frame. It also calculates the
%   average confidence score for these parts. Invalid points (with confidence
%   below the threshold) are excluded from the calculation by being treated
%   as NaNs.
%
%   Inputs:
%       - data: A 2D matrix containing the raw tracking data from DeepLabCut,
%         organized with [x, y, prob] triplets for each body part per frame.
%       - thr: A scalar confidence threshold. Tracking points below this
%         threshold are considered invalid and are not included in the
%         mean calculation.
%       - body_elements: A cell array of strings specifying the body parts to use
%         for calculating the center (e.g., {"swim_bladder", "tail_1"}).
%
%   Outputs:
%       - body_center: A 2xN matrix where the first row contains the calculated mean
%         x-positions and the second row contains the mean y-positions across the
%         specified body parts for each frame.
%       - avg_confidence: A 1xN vector containing the mean confidence score across
%         the specified body parts for each frame.
%
%   Example:
%       % Define the body parts used to find the center
%       body_center_parts = {"swim_bladder", "tail_1"};
%       thr = 0.9;
%
%       % Calculate the body center and average confidence
%       [center_coords, mean_conf] = get_body_center(data, thr, body_center_parts);
%
%   See also GET_VALID_XY.

if length(body_elements)>1
    part_xy = cellfun(@(x) get_valid_xy(data,body_parts,params,x,thr), body_elements,'UniformOutput',false);
    body_center = [mean(cell2mat(cellfun(@(x) x(1,:), part_xy, 'UniformOutput', false)'),'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x(2,:), part_xy, 'UniformOutput', false)'),'omitnan')];
    avg_confidence = mean(cell2mat(cellfun(@(x) data(get_parts_index(body_parts,params,x,'prob'),:), body_elements,'UniformOutput',false)'),'omitnan');
elseif isscalar(body_elements)
    % If only one body part is specified, get its valid coordinates directly.
    body_center = get_valid_xy(data,body_parts,params,body_elements,thr);
    avg_confidence = data(get_parts_index(body_parts,params,body_elements,'prob'),:);
end