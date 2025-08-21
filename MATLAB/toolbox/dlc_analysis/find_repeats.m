function output = find_repeats(input)
%FIND_REPEATS Finds consecutive repeated values in a numeric array.
%   OUTPUT = FIND_REPEATS(INPUT) calculates the number of consecutive
%   repetitions for each element in the input array. The output array has
%   the same size as the input, with each element indicating how many times
%   that value has been consecutively repeated up to that point.
%
%   This function is useful for identifying continuous blocks of identical
%   values, such as periods of no movement (zero velocity) in a tracking
%   trace.
%
%   Input:
%       - input: A 1xN numeric array.
%
%   Output:
%       - output: A 1xN array where each element contains the number of
%                 consecutive repetitions for the corresponding input value.
%
%   Example:
%       % Find the repeats in a sample array with alternating values
%       input_array = [1, 1, 1, 2, 2, 3, 3, 3, 3, 1];
%       repeat_counts = find_repeats(input_array);
%
%       % The output will be: [1, 2, 3, 1, 2, 1, 2, 3, 4, 1]
%
%   See also REPELEM, DIFF.

% TRUE if values change. A change is detected where the difference between
% adjacent elements is non-zero. A `true` is added at the start and end to
% capture the beginning and end of the entire array.
d = [true, diff(input) ~= 0, true];

% Number of repetitions for each block of identical values.
% `find(d)` gives the indices of the start of each block. `diff` calculates
% the length of each block.
n = diff(find(d));

% Replicate the block lengths to match the original array size.
% `repelem` repeats each element of `n` by `n` times, reconstructing the
% per-element repeat count for the entire array.
output = repelem(n, n);
end