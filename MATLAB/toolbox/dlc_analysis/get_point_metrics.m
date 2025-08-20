function point_metrics =get_point_metrics(body_part,varargin)
% GET_POINT_METRICS     Calculate point metrics 
%   [distance_mm, velocity, acceleration] = GET_POINT_METRICS(body_part)
%       Calculates distance travelled, velocity and acceleration in pixels
%       with time step of 1 (frames)
%
%   [distance_mm, velocity, acceleration] = GET_POINT_METRICS(body_part,**kwarg)
%       Calculates distance travelled, velocity and acceleration
%       Input is entered in style of ipython (see example below)
%
%       Input parameters
%       'tolerance'         | coverts upper limit of noise to zero
%       'avg_confidence'    | array input of tracking confidence to further filter noisy data points
%       'thr'               | threshold for confidence level (i.e. 0.9)
%       'frame_rate'        | sets frame rate so all calculation is converted into seconds (e.g. pixel/s or mm/s)
%       'pxl_resolution'    | sets resolution so all calculations is converted into mm (eg. mm/f or mm/s)
%       'smooth_val'        | size of sliding average window. Values must be odd numbered
%
%       Example:
%           [distance_mm, velocity, acceleration] =get_point_metrics(body_center,tolerance=0.3, smooth_val=5, frame_rate=300)
%
%           Uses body center to calculate distance, velocity and
%           acceleration. Values are in seconds, traces are smoothed with a
%           window of 5 frames, and any pixel distance values below 0.3 is
%           forced to zero.
%
%   [distance_mm, velocity, acceleration,distance_from_center] = GET_POINT_METRICS(body_part,**kwarg)
%       Calculates distance travelled, velocity, acceleration, distance
%       from center of well (pixel/mm)
%
%       Required Input
%       'frame_resolution'  | 1x2 array of pixel value
%
%   [distance_mm, velocity, acceleration,distance_from_center, normalized_from_center] = GET_POINT_METRICS(body_part,**kwarg)
%       Calculates distance travelled, velocity, acceleration, distance
%       from center of well and normalized distance from center
%       
%       Center of well is set at 0, extending from -1 to 1 from left to
%       right and -1 to 1 from bottom to top.


% Parse out varargin inputs in the style of python
py_input = varargin;
[frame_rate,tolerance,pxl_resolution,frame_resolution,smooth_val,thr,avg_confidence] = python_kwarg(py_input,'frame_rate','tolerance','pxl_resolution','frame_resolution','smooth_val','thr','avg_confidence');

% Check and balances
if ~exist('body_part','var')
    error('Missing target body part to calculate point metrics')
end
if ~isempty(thr)
    if isempty(avg_confidence)        
        error('Threshold provided but missing tracking confidence')
    end
else
    if ~isempty(avg_confidence)
        error('No confidence threhold provided')
    end
end
if (nargin-1)/2 > sum([py_input{:}]=='frame_rate' | [py_input{:}]=='tolerance' | [py_input{:}]=='pxl_resolution' | [py_input{:}]=='smooth_val' | [py_input{:}]=='thr' | [py_input{:}]=='avg_confidence' | [py_input{:}]=='frame_resolution')
    error('Unknown input variable name - Check spelling')
end
if nargout>5
    error('Too many outputs')
end

% Set Variables if none is provided
if isempty(frame_rate)
    frame_rate = 1;
end
if isempty(tolerance)
    tolerance = 0;
end
if isempty(pxl_resolution)
    pxl_resolution = 1;
end

% Smoothing for bout detection
if ~isempty(smooth_val)
    processed_array = [conv(body_part(1,:)',ones(smooth_val,1),'valid')/smooth_val conv(body_part(2,:)',ones(smooth_val,1),'valid')/smooth_val];
    processed_array = [nan((smooth_val-1)/2,2); processed_array; nan((smooth_val-1)/2,2)];
else
    processed_array = body_part';
end

% Remove frames where the average tracking confidence was low
if ~isempty(avg_confidence)
    valid_body_points = processed_array;
    valid_body_points(avg_confidence<thr,:)=NaN;
else
    valid_body_points = processed_array;
end

% Calulated distance/velocity/acceleration
distance = arrayfun(@(i) sqrt(((valid_body_points(i+1,1)-valid_body_points(i,1))^2)+((valid_body_points(i+1,2)-valid_body_points(i,2))^2)), 1:size(valid_body_points,1)-1);
distance_mm = distance.*pxl_resolution;
distance_mm(distance_mm<tolerance)=0;
distance_mm(distance_mm>1.85) = NaN;
velocity = distance_mm/(1/frame_rate);
acceleration = diff(distance_mm)/(1/frame_rate);
jerk = diff(velocity)/(1/frame_rate);


% Calculated distance from center
if ~isempty(frame_resolution)
        distance_from_center = sqrt((valid_body_points(:,1)-(frame_resolution(1)/2)).^2 + (valid_body_points(:,2)-(frame_resolution(2)/2)).^2);
        normalized_from_center = [(valid_body_points(:,1)-(frame_resolution(1)/2))/(frame_resolution(1)/2) ...
                        (valid_body_points(:,2)-(frame_resolution(2)/2))/(frame_resolution(2)/2)];
else
    distance_from_center = [];
    normalized_from_center = [];
end

% Output as structure
point_metrics.distance_mm = distance_mm;
point_metrics.velocity = velocity;
point_metrics.acceleration = acceleration;
point_metrics.jerk = jerk;
point_metrics.distance_from_center = distance_from_center;
point_metrics.normalized_from_center = normalized_from_center;


