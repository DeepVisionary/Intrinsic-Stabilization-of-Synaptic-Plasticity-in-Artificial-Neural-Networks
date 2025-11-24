
% Define parameters
inputSize = 2;
hiddenSize = 100;
outputSize = 1;
numSequences = 20;
numTestSequences = 20;
sequenceLength = 1;
numEpochs = 15000;

supervised_rate = 0.01;
topdown_rate = 0.001;

% Generate XOR data for training
[x1, x2, Y] = generate2bitparity(numSequences);
inputs = [x1, x2];
X = zeros(numSequences, sequenceLength, inputSize);
for i = 1:numSequences
    for j = 1:sequenceLength
        for k = 1:inputSize
            X(i, j, k) = inputs(i, k);
        end
    end
end

% Generate XOR data for testing
[x1, x2, Y_test] = generate2bitparity(numTestSequences);
inputs = [x1, x2];
X_test = zeros(numTestSequences, sequenceLength, inputSize);
for i = 1:numTestSequences
    for j = 1:sequenceLength
        for k = 1:inputSize
            X_test(i, j, k) = inputs(i, k);
        end
    end
end

% Initialize weights
Wxh = randn(hiddenSize, inputSize) * 0.01;
Whh = randn(hiddenSize, hiddenSize) * 0.01;
Why = randn(outputSize, hiddenSize) * 0.01;
bh = zeros(hiddenSize, 1);
by = zeros(outputSize, 1);

% Top-down node parameters
w_td = randn(1) * 0.01;
b_td = 0;

% Training loop
avgLoss = zeros(numEpochs, 1);
avgError = zeros(numEpochs, 1);
avg_pc_error = zeros(numEpochs, 1);
avgTopDownLoss = zeros(numEpochs, 1);
for epoch = 1:numEpochs
    totalLoss = 0;
    totalError = 0;
    total_pc_err = 0;
    totalTDLoss = 0;

    for seq = 1:numSequences
        % Initialize hidden state
        h = zeros(hiddenSize, 1);
        h_states = zeros(hiddenSize, sequenceLength);

        % Forward pass
        for t = 1:sequenceLength
            x_t = squeeze(X(seq, t, :));
            if size(x_t, 1) == 1, x_t = x_t'; end
            h = tanh(Wxh * x_t + Whh * h + bh);
            h_states(:, t) = h;
        end

        % Output layer
        y_pred = sigmoid(Why * h + by);

        % Losses
        error = mean((Y(seq) - y_pred).^2);
        totalError = totalError + error;
        y_thresh = single(y_pred >= 0.5);
        pc_err = single(y_thresh ~= Y(seq));
        total_pc_err = total_pc_err + pc_err;
        loss = - (Y(seq) * log(y_pred) + (1 - Y(seq)) * log(1 - y_pred));
        totalLoss = totalLoss + loss;

        % === Top-Down Node ===
        z_td = w_td * y_pred + b_td;
        a_td = sigmoid(z_td);
        J_td = 0.5 * (a_td - y_pred)^2;
        totalTDLoss = totalTDLoss + J_td;

        % === Gradients ===
        % Output gradients
        dL_dy = y_pred - Y(seq);
        dL_dWhy = dL_dy * h';
        dL_dby = dL_dy;

        % Top-down gradients (output layer only)
        dJ_dztd = (a_td - y_pred) * a_td * (1 - a_td);    % ∂J/∂z_td
        dJ_td_Why = dJ_dztd * w_td * y_pred * (1 - y_pred) * h';
        dJ_td_by = dJ_dztd * w_td * y_pred * (1 - y_pred);

        % Backpropagate through time (BPTT)
        dh = Why' * dL_dy;
        for t = sequenceLength:-1:1
            h_t = h_states(:, t);
            h_prev = zeros(hiddenSize, 1);
            if t > 1
                h_prev = h_states(:, t - 1);
            end
            dh = dh .* (1 - h_t.^2);
            dL_dWhh = dh * h_prev';
            dL_dWxh = dh * x_t';
            dL_dbh = dh;
            dh = Whh' * dh;
        end

        % === Parameter updates ===
        Wxh = Wxh - supervised_rate * dL_dWxh;
        Whh = Whh - supervised_rate * dL_dWhh;
        bh = bh - supervised_rate * dL_dbh;
        Why = Why - supervised_rate * dL_dWhy + topdown_rate * dJ_td_Why;
        by = by - supervised_rate * dL_dby + topdown_rate * dJ_td_by;

        % Update top-down parameters
        w_td = w_td - supervised_rate * (dJ_dztd * y_pred);
        b_td = b_td - supervised_rate * dJ_dztd;
    end

    avgLoss(epoch) = totalLoss / numSequences;
    avgError(epoch) = totalError / numSequences;
    avg_pc_error(epoch) = 100 * (total_pc_err / numSequences);
    avgTopDownLoss(epoch) = totalTDLoss / numSequences;

    if mod(epoch, 1000) == 0
        disp(['Epoch: ' num2str(epoch) ...
            ', Loss: ' num2str(avgLoss(epoch)) ...
            ', TD Loss: ' num2str(avgTopDownLoss(epoch)) ...
            ', Error: ' num2str(avg_pc_error(epoch)) '%']);
    end
end

% Test
totalTestError = 0;
save_y_pred = zeros(numTestSequences, 1);
for seq = 1:numTestSequences
    h = zeros(hiddenSize, 1);
    for t = 1:sequenceLength
        x_t = squeeze(X_test(seq, t, :));
        if size(x_t, 1) == 1, x_t = x_t'; end
        h = tanh(Wxh * x_t + Whh * h + bh);
    end
    y_pred = sigmoid(Why * h + by);
    save_y_pred(seq) = y_pred;
    totalTestError = totalTestError + mean((Y_test(seq) - y_pred).^2);
end


function s = sigmoid(x)
s = 1 ./ (1 + exp(-x));
end
