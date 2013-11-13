function X = physiological_features( record, outfile, winsize )
% function physiological_features( infile, outfile, winsize)
%
% Generate a physiological features using a sliding window over a
% continuous time series. Output these features to a text file. Currently
% DFA, sample entropy, sample asymmetry, and the  standard deviation are
% calculated for each window.
%
% Input:
%   record string file to calculate the features from, expects mimic2wdb
%   record
%   outfile string file to write the new features to
%   winsize int the number of samples to include in the window
%
% Output:
%   X (N-winsize) x (num feats) matrix of features, N is number of samples
%

% % Generate a qrs annotation file for the record
% sqrs(record);
% 
% % Compute IHR for the record using qrs annotation file
% [hr] = tach(record, 'qrs');
rec = fopen(record,'r');
[hr] = textscan(rec, '%f');
if size(hr{1},1) == 0
    return;
else
    hr = hr{1};
end

% Feature vector
numfeats = 4;
N = size(hr, 1); % number of samples
X = zeros(N-winsize+1, numfeats);
%outf = fopen(outfile, 'w');

% % Sweep a sliding window through the data
% for i=1:N-winsize,
%     endind = i + winsize;
%     fprintf('Computing window %d of %d\n', i, N-winsize); 
%     % sqrt of 2nd moment
%     sd = std(hr(i:endind,1));
%     % DFA
%     [slow, fast] = dfapeng(hr(i:endind,1));
%     % Sample asymmetry
%     [R1, R2] = smAsymm(hr(i:endind,1));
%     % Sample entropy
%     % Max template length 5 and tolerance .2 are hand chosen, but have been
%     % used historically and shown to work well for HRV
%     [e, A, B] = sampenc(hr(i:endind,1), 5, .2);
%     
%     X(i,:) = [sd, slow, fast, R1, R2, e'];
%     
%     fprintf(outf, '%f %f %f %f %f %f %f %f %f %f\n', X(i,:));
%     
% end
tic;
fprintf('Making windowed data\n');
HRwin = num2cell(hr(hankel(1:winsize,winsize:N)),1);
toc; fprintf('Made %d windows of size %d \n', size(HRwin,2), winsize);
toc;fprintf('Computing DFA features\n');
%[SL, FT] = cellfun(@(x) dfapeng(x),HRwin,'UniformOutput',false);
[alphas, ints, flucts] = cellfun(@(x) fastdfa(x),HRwin,'UniformOutput',false);
toc; fprintf('Computing SD features\n');
SD = cell2mat(cellfun(@(x) std(x),HRwin,'UniformOutput',false));
toc; fprintf('Computing sample asymmetry features\n');
[R1, R2] = cellfun(@(x) smAsymm(x),HRwin,'UniformOutput',false);
toc; fprintf('Computing sample entropy features\n'); 
E = cellfun(@(x) sampenc(x, 5, .2),HRwin,'UniformOutput',false);
toc;


X(:,1) = SD';
X(:,2) = cell2mat(alphas)';
X(:,3) = cell2mat(R1)';
X(:,4) = cell2mat(R2)';
X(:,5:9) = cell2mat(E)';
toc;
fprintf('Writing out file: %s\n', outfile);
dlmwrite(outfile, X, ' ');
toc;

    
