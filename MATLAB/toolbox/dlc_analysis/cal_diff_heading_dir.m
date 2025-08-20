function z = cal_diff_heading_dir(x,y)
%CAL_DIFF_HEADING_DIR Calculates the signed change in heading, accounting for 360-degree wrap-around.
%   Z = CAL_DIFF_HEADING_DIR(X, Y) computes the shortest-path signed difference
%   between two heading angles, X (initial angle) and Y (final angle). The
%   function determines if the change in direction is clockwise (positive) or
%   anti-clockwise (negative), correctly handling transitions across the 0/360
%   degree boundary.
%
%   Inputs:
%       - x: The heading angle at the initial time point (in degrees).
%       - y: The heading angle at the final time point (in degrees).
%
%   Output:
%       - z: The signed change in heading (in degrees). A positive value
%         indicates a clockwise turn, and a negative value indicates an
%         anti-clockwise turn.
%
%   Example:
%       % A clockwise turn from 350 to 10 degrees
%       diff1 = cal_diff_heading_dir(350, 10); % Returns 20
%
%       % An anti-clockwise turn from 10 to 350 degrees
%       diff2 = cal_diff_heading_dir(10, 350); % Returns -20
%
%       % A simple clockwise turn
%       diff3 = cal_diff_heading_dir(45, 90);  % Returns 45
%
%   See also GET_VECTOR_METRICS.

if abs(x-y) <=180
    if x<=y 
        z = abs(x-y);   % if the first is smaller, that means it going clockwise (therefore positive)
    else 
        z = y-x;    % if the first is bigger, that means its going anticlockwise (therefore negative)
    end
else                % Means the smallest angle is crossing the origin (0 degrees)
    if x<y 
        z = abs(x-y) - 360;     %if the first is smaller, means it is going clockwise - so subtract 360
    else
        z = 360 - abs(x-y);     %if the first is bigger, means it is going anticlockwise - so 360 subtract the value (same as mutliplying by - 1)
    end
end

