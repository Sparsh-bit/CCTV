import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_seq_length: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[..., :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class TransformerBehaviorPredictor(nn.Module):

    def __init__(self, input_dim: int = 6, d_model: int = 256, num_heads: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1,
                 output_dim: int = 1, max_seq_length: int = 1000):

        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
            nn.Sigmoid()
        )

        self.attention_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.input_proj(x)

        x = self.pos_encoder(x)

        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        attention_weights = torch.softmax(self.attention_head(x), dim=1)

        x_agg = (x * attention_weights).sum(dim=1)

        anomaly_scores = self.anomaly_head(x_agg)

        return anomaly_scores, attention_weights

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = Q.shape[0]

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(context)

        return output, attention_weights

class ConvLSTMBehaviorDetector(nn.Module):

    def __init__(self, input_channels: int = 3, hidden_channels: int = 64,
                 num_layers: int = 3, kernel_size: int = 3, dropout: float = 0.1,
                 output_dim: int = 1):

        super().__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.conv_input = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )

        self.convlstm_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels if i == 0 else hidden_channels
            self.convlstm_layers.append(
                ConvLSTMCell(in_ch, hidden_channels, kernel_size)
            )

        self.dropout = nn.Dropout(dropout)

        self.anomaly_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len, _, height, width = x.shape

        h = self.conv_input(x[:, 0])

        h_states = [torch.zeros(batch_size, self.hidden_channels, h.shape[2], h.shape[3],
                               device=x.device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(batch_size, self.hidden_channels, h.shape[2], h.shape[3],
                               device=x.device) for _ in range(self.num_layers)]

        attention_maps = []

        for t in range(seq_len):
            if t > 0:
                h = self.conv_input(x[:, t])

            for i, convlstm in enumerate(self.convlstm_layers):
                h, c = convlstm(h, h_states[i], c_states[i])
                h_states[i] = h
                c_states[i] = c

            attention_maps.append(h.mean(dim=1, keepdim=True))

        attention_maps = torch.cat(attention_maps, dim=1)

        anomaly_scores = self.anomaly_head(h_states[-1])

        return anomaly_scores, attention_maps

class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_channels + hidden_channels, 4 * hidden_channels,
            kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        combined = torch.cat([x, h], dim=1)
        i, f, g, o = torch.split(self.conv(combined), self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new

if __name__ == "__main__":

    model = TransformerBehaviorPredictor(input_dim=6, d_model=256, num_heads=8)
    x = torch.randn(4, 30, 6)

    anomaly_scores, attention = model(x)
    print(f"Anomaly scores: {anomaly_scores.shape}")
    print(f"Attention weights: {attention.shape}")

    convlstm = ConvLSTMBehaviorDetector(input_channels=3, hidden_channels=64)
    x_video = torch.randn(2, 10, 3, 224, 224)

    scores, attn = convlstm(x_video)
    print(f"ConvLSTM scores: {scores.shape}")
    print(f"ConvLSTM attention: {attn.shape}")
