function D = make_demo_admissible_solutions()
% ============================================================
% Synthetic admissible-solution ensemble for schematic Fig. e
%
% This is VISUAL DEMONSTRATION DATA ONLY.
% It does not represent actual fitted model results.
%
% Outputs:
%   D.t          : time
%   D.baseline1  : baseline blue trajectory
%   D.baseline2  : baseline green trajectory
%   D.sol1       : N x T blue trajectories
%   D.sol2       : N x T green trajectories
%   D.distance   : N x N standardized functional distance
% ============================================================

rng(20260915);

N  = 100;
nt = 401;
t  = linspace(0,1,nt);

%% ------------------------------------------------------------
% Baseline schematic trajectories
% ------------------------------------------------------------

baseline1 = ...
    0.78*exp(-0.5*((t-0.43)/0.15).^2) + ...
    0.10*exp(-0.5*((t-0.72)/0.22).^2);

baseline2 = ...
    0.66*exp(-0.5*((t-0.35)/0.17).^2) + ...
    0.24*exp(-0.5*((t-0.72)/0.16).^2);

baseline1 = baseline1 + 0.025;
baseline2 = baseline2 + 0.020;

%% ------------------------------------------------------------
% Generate diverse but smooth trajectories
% ------------------------------------------------------------

sol1 = zeros(N,nt);
sol2 = zeros(N,nt);

for k = 1:N

    % -------- blue family --------
    shift1 = 0.065*randn;
    amp1   = 0.72 + 0.32*rand;
    width1 = 0.115 + 0.065*rand;

    shoulderLoc1 = 0.58 + 0.23*rand;
    shoulderA1   = 0.05 + 0.28*rand;
    shoulderW1   = 0.10 + 0.10*rand;

    y1 = ...
        amp1 .* exp(-0.5*((t-(0.42+shift1))/width1).^2) + ...
        shoulderA1 .* exp(-0.5*((t-shoulderLoc1)/shoulderW1).^2);

    % Smooth low-frequency deformation
    phase1 = 2*pi*rand;
    deform1 = ...
        0.035*sin(2*pi*t + phase1) + ...
        0.018*sin(4*pi*t + 0.7*phase1);

    y1 = y1 .* (1 + deform1);
    y1 = y1 + 0.015 + 0.018*rand;

    % -------- green family --------
    shift2 = 0.075*randn;
    amp2   = 0.58 + 0.34*rand;
    width2 = 0.125 + 0.075*rand;

    shoulderLoc2 = 0.61 + 0.22*rand;
    shoulderA2   = 0.08 + 0.32*rand;
    shoulderW2   = 0.09 + 0.11*rand;

    y2 = ...
        amp2 .* exp(-0.5*((t-(0.35+shift2))/width2).^2) + ...
        shoulderA2 .* exp(-0.5*((t-shoulderLoc2)/shoulderW2).^2);

    phase2 = 2*pi*rand;
    deform2 = ...
        0.040*sin(2*pi*t + phase2) + ...
        0.020*sin(4*pi*t + 0.5*phase2);

    y2 = y2 .* (1 + deform2);
    y2 = y2 + 0.012 + 0.018*rand;

    sol1(k,:) = max(y1,0);
    sol2(k,:) = max(y2,0);
end

%% ------------------------------------------------------------
% Pairwise standardized functional distance
%
% Treat the two schematic trajectories jointly.
% This follows the spirit of the paper's functional distance,
% but is ONLY for graphical demonstration.
% ------------------------------------------------------------

% Standardize each family by its total ensemble range
range1 = max(sol1(:)) - min(sol1(:));
range2 = max(sol2(:)) - min(sol2(:));

Z1 = sol1 / range1;
Z2 = sol2 / range2;

Dist = zeros(N,N);

for i = 1:N
    for j = i+1:N

        d1 = sqrt(mean((Z1(i,:) - Z1(j,:)).^2));
        d2 = sqrt(mean((Z2(i,:) - Z2(j,:)).^2));

        dij = 0.5*(d1+d2);

        Dist(i,j) = dij;
        Dist(j,i) = dij;
    end
end

%% Output
D.t         = t;
D.N         = N;

D.baseline1 = baseline1;
D.baseline2 = baseline2;

D.sol1      = sol1;
D.sol2      = sol2;

D.distance  = Dist;

end