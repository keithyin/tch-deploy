import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bigru = nn.GRU(
            10, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True
        )
        self.linear = nn.Linear(in_features=2 * 128, out_features=5, bias=False)

    def forward(self, batch_feature, batch_length):
        x = nn.utils.rnn.pack_padded_sequence(
            batch_feature, batch_length, True, enforce_sorted=False
        )
        x, _ = self.bigru(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.sum(dim=1)
        x = self.linear(x)
        x = torch.softmax(x, dim=1)
        return x


if __name__ == "__main__":
    model = Model()
    example = torch.rand(2, 10, 10)
    example_lengths = torch.tensor([8, 10])

    # 跟踪模型
    traced_script_module = torch.jit.trace(model, [example, example_lengths])

    # 保存模型
    traced_script_module.save("traced_model.pt")
