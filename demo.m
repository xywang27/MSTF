%% This is the MATLAB code for the following paper:

%   Multiplicative Sparse Tensor Factorization for Multi-View Multi-Task Learning

%   Please run 'demo.m' to demostrate the training of vMSTF and MSTF on the synthetic data set .

%%

clear;

rng('default');

%% load data 

load_data = load('./synthetic.mat'); 

data = load_data.X_cell;

target = load_data.y_cell;

designedW = load_data.W;

%% Set optimization parameters for vMSTF

opts.p = 2;                       % p = {1,2}

opts.k = 1;                       % k = {1,2}

opts.max_iter = 1e3;              % number of max iterations

opts.rel_tol = 1e-4;              % termination condition
  
hyp = [1e3,1];                    % [lambda1, lambda2]

%% Build vMSTF model using the optimal parameter 

[learnedW_vMSTF,~, ~, ~] = vMSTF(data,target,hyp,opts);

%% Set optimization parameters for MSTF

opts.p = 2;                       % p = {1,2}

opts.k = 2;                       % k = {1,2}

opts.max_iter = 1e3;              % number of max iterations

opts.rel_tol = 1e-4;              % termination condition

hyp = [1e-1,1e-1,1e-1,0.2];            % [lambda1, lambda2, lambda3, alpha(latent factor ratio)]


%% Build MSTF model using the optimal parameter 

[learnedW_MSTF,~,~,~,~,~] = MSTF(data,target,hyp,opts);

%% Illustrate results

subplot(1,3,1)
h1 = heatmap(-abs(double(designedW(:,:,1))));
h1.XDisplayLabels = nan(size(h1.XDisplayLabels));
h1.YDisplayLabels = nan(size(h1.YDisplayLabels));
h1.Colormap = gray;
h1.GridVisible = 'off';
h1.ColorbarVisible = 'off';
h1.ColorLimits = [-10, 0];
h1.Title = 'Designed W_{::1}^*';

subplot(1,3,2)
h2 = heatmap(-abs(reshape(learnedW_vMSTF(:,1),[20,20])));
h2.XDisplayLabels = nan(size(h2.XDisplayLabels));
h2.YDisplayLabels = nan(size(h2.YDisplayLabels));
h2.Colormap = gray;
h2.GridVisible = 'off';
h2.ColorbarVisible = 'off';
h2.ColorLimits = [-10, 0];
h2.Title = ' W_{::1} by vMSTF';

subplot(1,3,3)
h3 = heatmap(-abs(double(learnedW_MSTF{1})));
h3.XDisplayLabels = nan(size(h3.XDisplayLabels));
h3.YDisplayLabels = nan(size(h3.YDisplayLabels));
h3.Colormap = gray;
h3.GridVisible = 'off';
h3.ColorbarVisible = 'off';
h3.ColorLimits = [-10, 0];
h3.Title = ' W_{::1} by MSTF';
