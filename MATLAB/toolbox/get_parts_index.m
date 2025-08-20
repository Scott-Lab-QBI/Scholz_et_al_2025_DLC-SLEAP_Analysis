function index = get_parts_index(body_parts, params, body,par)

%GET_PARTS_INDEX Get the linear index for a specific body part and parameter.
%   INDEX = GET_PARTS_INDEX(BODY_PARTS, PARAMS, BODY, PAR) calculates the 
%   linear index for a specific combination of a body part and a parameter. 
%   This is useful for accessing a flattened data array where a group of 
%   parameters (e.g., x, y, probability) are stored contiguously for each 
%   body part.
%
%   Inputs:
%       - body_parts: A cell array or string array containing the names of 
%         all tracked body parts (e.g., {"tail_1", "R_eye"}).
%       - params: A cell array or string array containing the parameter 
%         names for each body part (e.g., {"x", "y", "prob"}).
%       - body: A string specifying the desired body part name (e.g., "tail_1").
%       - par: A string specifying the desired parameter name (e.g., "y").
%
%   Output:
%       - index: The calculated linear index corresponding to the specified 
%         body part and parameter.
%
%   Example:
%       body_parts = ["tail_1", "R_eye"];
%       params = ["x", "y", "prob"];
%       
%       % Get the index for the 'y' parameter of 'R_eye'
%       idx = get_parts_index(body_parts, params, "R_eye", "y");
%       
%   See also FIND.

which_body_ind = find(body_parts == body);
which_param_ind = find(params == par);
index = (which_body_ind-1)*3+which_param_ind;

