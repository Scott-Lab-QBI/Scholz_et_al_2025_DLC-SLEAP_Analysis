function varargout = python_kwarg(py_input,varargin)
%PYTHON_KWARG Extracts values from a key-value pair cell array.
%   [VAL1, VAL2, ...] = PYTHON_KWARG(PY_INPUT, KEY1, KEY2, ...) extracts
%   values associated with specified keys from a cell array that stores
%   alternating key-value pairs, similar to a Python keyword arguments dictionary.
%
%   This function is useful for parsing arguments passed to a function
%   where the inputs are not fixed in number or order.
%
%   Inputs:
%       - py_input: A cell array containing alternating key-value pairs.
%         For example, {'name', 'john', 'age', 30, 'city', 'new york'}.
%         The keys must be string or character arrays.
%
%       - varargin: A variable number of strings (character or string arrays)
%         representing the keys whose values you wish to retrieve.
%
%   Outputs:
%       - varargout: A variable number of outputs, where each output
%         corresponds to the value of the requested key. The order of the
%         outputs matches the order of the input keys. If a key is not
%         found, the corresponding output variable will be an empty cell.
%
%   Example:
%       % Assume you have a cell array of key-value pairs
%       py_args = {'name', 'Alice', 'score', 95, 'is_active', true};
%
%       % Retrieve the 'score' and 'is_active' values
%       [score, active] = python_kwarg(py_args, 'score', 'is_active');
%
%       % The values are now stored in the variables 'score' (95) and
%       % 'active' (true).
%       fprintf('Score: %d, Is Active: %d\n', score, active);
%
%   See also PARSEARGS.

nOutputs = nargout;
nInputs = nargin;

varargout = cell(1,nOutputs);

input_var_str = py_input(cellfun(@(x) isa(x,'string'),py_input));
for i = 1:nInputs-1
    var_str = varargin{i};
    if sum(([py_input{:}]==var_str))==1
        varargout{i} = py_input{find([input_var_str{:}]==var_str)*2};
    end
end