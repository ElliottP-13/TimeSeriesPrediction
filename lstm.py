import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils import data
from tqdm import tqdm


class LSTMTest(nn.Module):
    def __init__(self):
        super(LSTMTest, self).__init__()

        input_dim = 1
        self.lstm = nn.LSTM(input_dim, hidden_size=12, num_layers=3, batch_first=True)
        self.fc = nn.Linear(12, 1)

    def forward(self, x, h0=None, c0=None, time_predictions=False):
        batched = x.ndim == 3
        # if not batched:
        #     x = x.unsqueeze(0)
        if h0 is None and c0 is None:
            lstm_out, (hn, cn) = self.lstm(x, None)  # none inits the h0,c0 to zero
        else:
            lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        if batched:
            reshape = hn[-1].view(-1, 12)  # get the last hidden state (from last layer)
        else:
            reshape = hn[-1].view(12)
        out = self.fc(reshape)

        if time_predictions:  # return a prediction for every point in time
            time_preds = self.fc(lstm_out)
            return out, (hn, cn), time_preds

        return out, (hn, cn)

    def n_step(self, last_prediction, hn, cn, future_steps):
        outputs = []
        with torch.no_grad():
            for i in range(future_steps):
                out, (hn, cn) = self.forward(last_prediction.unsqueeze(last_prediction.ndim - 1), hn, cn)
                outputs.append(out)
                last_prediction = out
        return outputs

    def test_and_plot(self, x_test, y_test):
        f = plt.figure()
        o, (h, c), time_pred = self.forward(x_test, time_predictions=True)
        n_steps = 100
        preds = self.n_step(o, h, c, n_steps)

        x_plt = x_test.squeeze(-1).tolist()
        target_plt = y_test.tolist()
        output_plt = time_pred.squeeze(-1).tolist()

        x_plt2 = list(np.arange(n_steps) + len(x_test))
        y_plt2 = []
        for t in preds:
            y_plt2.append(t.item())

        plt.plot(x_plt, target_plt, 'k-', label='Target')
        plt.plot(x_plt, output_plt, 'b-', label='Output')
        plt.plot(x_plt2, y_plt2, 'r--', label='Future')
        plt.legend()
        return f


def create_train_data(x: torch.Tensor, y: torch.Tensor, window_size, stride=1, single_y=True):
    """
    B = number of sequences
    L = sequence length
    F = number features
    :param x: shape B,L,F if batched, else L,F
    :param y: shape B,L if batched, else L
    :param window_size: 
    :param stride: 
    :return: x with shape: N,window_size,F; y with shape: N,1
    """
    batched = x.ndim == 3

    x_out = x.unfold(x.ndim - 2, window_size, stride)
    x_out = x_out.swapaxes(x_out.ndim - 2, x_out.ndim - 1)

    y2 = y.unfold(x.ndim - 2, window_size, stride)

    if batched:
        y_out = y2[:, :, :, -1] if single_y else y2.swapaxes(y2.ndim - 2, y2.ndim - 1)
        y_out = y_out.flatten(0, 1)
        x_out = x_out.flatten(0, 1)
    else:
        y_out = y2[:, :, -1] if single_y else y2.swapaxes(y2.ndim - 2, y2.ndim - 1)

    return x_out, y_out


if __name__ == '__main__':
    N = 100  # number of samples
    L = 1000  # length of each sample (number of values for each sine wave)
    T = 20  # width of the wave
    x = np.empty((N, L), np.float32)  # instantiate empty array
    x[:] = np.arange(L) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    y = np.sin(x / 1.0 / T).astype(np.float32)

    x = torch.tensor(np.arange(L) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1), dtype=torch.float32).unsqueeze(-1)
    y = torch.sin(x / T)

    lstm = LSTMTest()
    o, (h, c), preds = lstm.forward(x, time_predictions=True)

    x_o, y_o = create_train_data(x, y, 300, 1, False)

    # used for checking non-batched inputs to creat_train_data
    # x_test = torch.tensor(np.arange(1000), dtype=torch.float32).unsqueeze(-1)
    # y_test = torch.sin(x_test / T)
    # x_o, y_o = create_train_data(x_test, y_test, 300, 1, False)

    dataset = data.TensorDataset(x_o, y_o)
    train_loader = data.DataLoader(dataset, shuffle=True, batch_size=256)

    epochs = 15
    loss_fn = nn.MSELoss()
    optimizer = optim.LBFGS(lstm.parameters(), lr=0.08)
    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            inputs, targets = batch

            def closure():
                optimizer.zero_grad()
                single_step, _, multi_step = lstm(inputs, time_predictions=True)
                loss = loss_fn(multi_step, targets)
                loss.backward()
                return loss

            optimizer.step(closure)

        x_test = torch.tensor(np.arange(50), dtype=torch.float32).unsqueeze(-1)
        y_test = torch.sin(x_test / T)

        fig = lstm.test_and_plot(x_test, y_test)
        fig.suptitle(f"Epoch {epoch}")
        plt.show()
