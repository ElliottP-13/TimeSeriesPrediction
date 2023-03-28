import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils import data
from tqdm import tqdm


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class LTSFBase(nn.Module):
    def __init__(self, L, T, in_channel, out_channel):
        super(LTSFBase, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.L = L
        self.T = T

    def n_step(self, x, n_steps=100):
        # x: [Batch, Input length, Channel] or [Input length, Channel]
        if x.ndim == 2:
            x = x.unsqueeze(0)  # add batch dimension
        chunks = n_steps // self.T  # get number of jumps
        # TODO maybe make sliding window, so make repeated guesses at stride length and use avg?

        outputs = []
        xp = torch.clone(x)
        for i in range(chunks):
            pred = self.forward(xp)
            xp = torch.cat((xp, pred.unsqueeze(-1)), 1)  # append predictions
            xp = xp[:, -self.L:, :]  # keep last window_size

            outputs.append(pred)

        return torch.cat(outputs, 1)[0, :n_steps]  # stick them all together into [Batch, n_steps, channel]

    def test_and_plot(self, x_test, y_test, n_steps=100):
        f = plt.figure()

        preds = self.n_step(x_test, n_steps)

        x_plt = list(np.arange(len(x_test)))
        target_plt = y_test.tolist()

        x_plt2 = list(np.arange(n_steps) + len(x_test))
        y_plt2 = []
        for t in preds:
            y_plt2.append(t.item())

        plt.plot(x_plt, target_plt, 'k-', label='Input')
        plt.plot(x_plt2, y_plt2, 'r--', label='Future')
        plt.legend()
        return f


class LTSFLinear(LTSFBase):
    def __init__(self, L, T, in_channel, out_channel):
        super(LTSFLinear, self).__init__(L, T, in_channel, out_channel)

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.L = L
        self.T = T

        L = L * in_channel  # for flattening input channels

        if out_channel == 1:
            self.layer = nn.Linear(L, T)
        else:
            self.layer = nn.ModuleList()
            for i in range(self.out_channel):
                self.layer.append(nn.Linear(L, T))

    def forward(self, x, batched=True):
        # x: [Batch, Input length, Channel]
        x = x.flatten(-2)  # flatten channel dimension away

        if self.out_channel == 1:
            return self.layer(x)
        else:
            return torch.stack([l(x) for l in self.layer], -1)  # stack in [Batch, input length, chanel]


class DLinear(LTSFBase):
    def __init__(self, L, T, in_channel, out_channel, kernel_size):
        super(DLinear, self).__init__(L, T, in_channel, out_channel)

        self.decomp = series_decomp(kernel_size)
        self.seasonal = LTSFLinear(L, T, in_channel, out_channel)
        self.trend = LTSFLinear(L, T, in_channel, out_channel)

    def forward(self, x):
        season, trend = self.decomp(x)

        season_out = self.seasonal(season)
        trend_out = self.trend(trend)

        return season_out + trend_out


class NLinear(LTSFBase):
    def __init__(self, L, T, in_channel, out_channel):
        super(NLinear, self).__init__(L, T, in_channel, out_channel)

        self.layer = LTSFLinear(L, T, in_channel, out_channel)

    def forward(self, x):
        last = x[:, -1, :].detach()
        x = x - last

def create_train_data2(x: torch.Tensor, y: torch.Tensor, L, T, stride=1):
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

    window_size = L + T

    x_out = x.unfold(x.ndim - 2, window_size, stride)
    x_out = x_out.swapaxes(x_out.ndim - 2, x_out.ndim - 1)

    y2 = y.unfold(x.ndim - 2, window_size, stride)

    if batched:
        y_out = y2.swapaxes(y2.ndim - 2, y2.ndim - 1)
        y_out = y_out.flatten(0, 1)
        x_out = x_out.flatten(0, 1)
        y_out = y_out[:, -T:, :]
        x_out = x_out[:, :L, :]
    else:
        y_out = y2.swapaxes(y2.ndim - 2, y2.ndim - 1)
        y_out = y_out[-T:, :]
        x_out = x_out[:L, :]

    return x_out, y_out

if __name__ == '__main__':
    from LTSF_Linear import *

    N = 100  # number of samples
    L = 1000  # length of each sample (number of values for each sine wave)
    T = 20  # width of the wave
    x = np.empty((N, L), np.float32)  # instantiate empty array
    x[:] = np.arange(L) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    y = np.sin(x / 1.0 / T).astype(np.float32)

    x = torch.tensor(np.arange(L) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1), dtype=torch.float32).unsqueeze(
        -1)
    y = torch.sin(x / T)

    L = 300
    Ti = 200

    model = LTSFLinear(L, Ti, 1, 1)

    x_o, y_o = create_train_data2(y, y, L, Ti, 1)

    dataset = data.TensorDataset(x_o, y_o.squeeze(-1))
    # dataset = data.TensorDataset(y[:, :-1], y[:, 1:])
    train_loader = data.DataLoader(dataset, shuffle=True, batch_size=256)

    epochs = 15
    loss_fn = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.08)
    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            inputs, targets = batch


            def closure():
                optimizer.zero_grad()
                out = model(inputs)
                loss = loss_fn(out, targets)
                loss.backward()
                return loss


            optimizer.step(closure)

        x_test = torch.tensor(np.arange(1000), dtype=torch.float32).unsqueeze(-1)
        y_test = torch.sin(x_test / T)

        fig = model.test_and_plot(y_test[-L:], y_test[-L:], n_steps=1000)
        fig.suptitle(f"Epoch {epoch}")
        plt.show()