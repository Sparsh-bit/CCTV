try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    class nn:  # type: ignore
        Module = object


class GCNConv(nn.Module if hasattr(nn, 'Module') else object):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, x, edge_index):
        return x


class GraphConv(GCNConv):
    pass


class TAGConv(GCNConv):
    pass


class GATConv(GCNConv):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0):
        super().__init__(in_channels, out_channels)
