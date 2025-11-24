
%predictive coding using ENMs

%2D inputs
N = 2;

alpha = 1; %leak term

% Parameters
T = 100; % Number of time steps
num_targets = 8; % Number of peripheral targets
radius = 5; % Radius of the movement (distance from center to targets)
center = [0; 0]; % Center location
curve_strength = 0.2; % Strength of the curvature (perpendicular offset)

% Generate target angles (8 evenly spaced directions)
angles = linspace(0, 2*pi, num_targets + 1); % Include 2*pi to close the loop
angles = angles(1:end-1); % Remove the last angle (duplicate of the first)

% Initialize X
X = zeros(2, T * num_targets * 2); % 2 (x, y) rows, times for each target (out + back)

% Generate trajectories
time = linspace(0, 1, T); % Normalized time for each segment
idx = 1;

for i = 1:num_targets
    % Calculate the target coordinates
    target = radius * [cos(angles(i)); sin(angles(i))];

    % Perpendicular direction for the curvature
    direction = target - center;
    perpendicular = [-direction(2); direction(1)];
    perpendicular = perpendicular / norm(perpendicular) * curve_strength;

    % Movement to the target (center to target with curve)
    for t = 1:T
        progress = time(t);
        curve = sin(pi * progress) * perpendicular; % Curvature changes smoothly
        X(:, idx) = center + (target - center) * progress + curve;
        idx = idx + 1;
    end

    % Movement back to the center (target to center with opposite curve)
    for t = 1:T
        progress = time(t);
        curve = sin(pi * progress) * (-perpendicular); % Reverse curvature
        X(:, idx) = target + (center - target) * progress + curve;
        idx = idx + 1;
    end
end

Y = X;

% Rotation angle in radians
rotation_angle = deg2rad(25);

% Rotation matrix
R = [cos(rotation_angle), -sin(rotation_angle);
    sin(rotation_angle),  cos(rotation_angle)];

% Apply the rotation to all points in X
X_rotated = R * X;

T = size(X,2);

M = 50; %default: 50
Z = randn(M,N);

W_in = randn(M);

dt=1;

%network activity time constant
tau = 0.1; %default: 0.1

%prediction update time constant
tau_b = 0.005; %default: 0.1

%supervised weight:
gamma = 0.1; %default: 0.5

%predictive weight:
sigma = 0; %default: 0.05

num_iter = 80;
%%%%%%%%%%%%%%%%%%%%%

I = Z*X;
K = zeros(M,T);
K(:,1) = I(:,1);
for t = 2:T/dt
    dK = -alpha.*K(:,t-1)+ReLU((K(:,t-1)'*W_in)')+I(:,t)+1;
    K(:,t) = K(:,t)+tau.*dK;
end

%scaling
Ks = scaledata(K,0,1/(M^2));

B = randn(N,T);
Bhat = randn(N,T);
c = zeros(num_iter,1);
c2 = zeros(num_iter,1);
sup_err = zeros(num_iter,1);
pred_err = zeros(num_iter,1);
W = randn(M,N);
save_B = zeros(num_iter,1);
save_Bhat = zeros(num_iter,1);
var_B = zeros(num_iter,1);
for i = 1:num_iter

    %basic rule
    W = (pinv(Ks'))*(gamma.*(Y-B))' - (pinv(Ks'))*(sigma.*(Bhat-B))'; %update weights

    %update prediction
    dBhat = -Bhat + B;
    Bhat = Bhat + tau_b*dBhat;

    %network output
    dB = -B + (Ks'*W)';
    B = B + tau*dB;

    sup_err(i) = mean((B(:)-Y(:)).^2);

    %early stopping
    if i>1
        if sup_err(i) > sup_err(i-1)
            sup_err(i:end) = sup_err(i-1);
            break;
        end
    end
end
