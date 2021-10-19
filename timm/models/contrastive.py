import torch.nn as nn
import torch.functional as F

featdim_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
}

class ContrastiveNet(nn.Module):
    """Contrastive model basic structure of backbone + project head -- SimCLR with same encoder f and g"""
    def __init__(self, encoder, name='resnet50', head='mlp', feat_dim=128):
        super(ContrastiveNet, self).__init__()
        self.encoder = encoder
        dim_in = featdim_dict[name]
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat