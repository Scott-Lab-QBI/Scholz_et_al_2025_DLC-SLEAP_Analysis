function output = quick_smooth(input,smooth_val)
%QUICK_SMOOTH Performs a fast moving average smoothing on an array.
%   OUTPUT = QUICK_SMOOTH(INPUT, SMOOTH_VAL) applies a moving average filter
%   to the input array using a convolution-based method. This is a fast
%   alternative to functions like `movmean` and is particularly efficient for
%   smoothing large data sets.
%
%   Inputs:
%       - input:        A numeric array (1xN or Nx1) to be smoothed.
%
%       - smooth_val:   An odd integer specifying the window size for the
%                       moving average. Using an odd number ensures that the
%                       smoothing window is centered on each data point.
%
%   Output:
%       - output:       The smoothed array. The length of the output will be
%                       `length(input) - smooth_val + 1` due to the
%                       'valid' convolution method, which discards
%                       edge effects.
%
%   Example:
%       % Create a sample noisy signal
%       noisy_signal = [1, 2, 3, 5, 4, 6, 8, 7, 10, 9];
%
%       % Smooth the signal with a window of 3
%       smoothed_signal = quick_smooth(noisy_signal, 3);
%
%       % The output will be: [2, 3.3333, 4, 5, 6, 7, 8.3333, 8.6667]
%
%   See also CONV, MOVSUM, MOVMEAN.

% `conv` performs the convolution of the input with a window of ones.
% The 'valid' flag ensures that the output is only the portion of the
% convolution where the window completely overlaps with the input array.
% This avoids padding the edges.
% The result is then divided by the window size to get the average.
output = conv(input, ones(smooth_val,1), 'valid')/smooth_val;
end