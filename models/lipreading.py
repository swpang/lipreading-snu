import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from abc import ABC
from models.base_model import Model
from models.resnet import ResNet, BasicBlock
from models.resnet1D import ResNet1D, BasicBlock1D
from models.shufflenetv2 import ShuffleNetV2
from models.senet import SENet, SEBasicBlock
from models.tcn import MultibranchTemporalConvNet, TemporalConvNet
from torch.utils.tensorboard import SummaryWriter
from utils.mixup import mixup_data, mixup_criterion
from utils.optim_utils import CosineScheduler, get_optimizer
from efficientnet_pytorch import EfficientNet


# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B):
    return torch.stack( [torch.mean( x[index][:,0:i], 1 ) for index, i in enumerate(lengths)],0 )


class MultiscaleMultibranchTCN(nn.Module):
    # MS-TCN implementation
    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len( self.kernel_sizes )

        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = x.transpose(1, 2)
        out = self.mb_ms_tcn(xtrans)
        out = self.consensus_func( out, lengths, B )
        return self.tcn_output(out)


class TCN(nn.Module):
    """Implements Temporal Convolutional Network (TCN)
    __https://arxiv.org/pdf/1803.01271.pdf
    """

    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(TCN, self).__init__()
        self.tcn_trunk = TemporalConvNet(input_size, num_channels, dropout=dropout, tcn_options=tcn_options, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

        self.has_aux_losses = False

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func( x, lengths, B )
        return self.tcn_output(x)


class Lipreading(Model, ABC):
    name = 'lipreading'
    def __init__(self, config, writer: SummaryWriter):
        Model.__init__(self, config, writer)
        self.modality = config['modality']
        self.num_classes = config['num_classes']
        self.hidden_dim = config['hidden_dim']
        self.backbone_type = config['backbone_type']
        self.extract_feats = config['extract_feats']
        self.width_mult = config['width_mult']
        self.relu_type = config['relu_type']
        self.tcn_options = config['tcn_options']
        self.criterion = nn.CrossEntropyLoss()

        if self.modality == 'raw_audio':
            self.frontend_nout = 1
            self.backend_out = 512
            self.trunk = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type=self.relu_type)
        elif self.modality == 'video':
            # we only use video data for our models
            if self.backbone_type == 'resnet':
                self.frontend_nout = 64
                self.backend_out = 512
                self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=self.relu_type)
            elif self.backbone_type == 'senet': 
                self.frontend_nout = 64
                self.backend_out = 512
                self.trunk = SENet(SEBasicBlock, [2, 2, 2, 2], relu_type=self.relu_type)
            elif self.backbone_type == 'shufflenet':
                assert self.width_mult in [0.5, 1.0, 1.5, 2.0], "Width multiplier not correct"
                shufflenet = ShuffleNetV2( input_size=96, width_mult=self.width_mult)
                self.trunk = nn.Sequential( shufflenet.features, shufflenet.conv_last, shufflenet.globalpool)
                self.frontend_nout = 24
                self.backend_out = 1024 if self.width_mult != 2.0 else 2048
                self.stage_out_channels = shufflenet.stage_out_channels[-1]
            elif self.backbone_type == 'efficientnet':
                self.trunk = EfficientNet.from_name('efficientnet-b3')
                self.frontend_nout = 3
                self.backend_out = 1000

            frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if self.relu_type == 'prelu' else nn.ReLU()
            # frontend includes 3D CNN, batch normalization, prelu activation, and max pooling
            self.frontend3D = nn.Sequential(
                        nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                        nn.BatchNorm3d(self.frontend_nout),
                        frontend_relu,
                        nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        else:
            raise NotImplementedError

        # all classes use multiscalemultibranchtcn (i.e., MS-TCN)
        tcn_class = TCN if len(self.tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
        self.tcn = tcn_class( input_size=self.backend_out,
                              num_channels=[
                                    self.hidden_dim * len(self.tcn_options['kernel_size']) * self.tcn_options['width_mult']
                                ] * self.tcn_options['num_layers'],
                              num_classes=self.num_classes,
                              tcn_options=self.tcn_options,
                              dropout=self.tcn_options['dropout'],
                              relu_type=self.relu_type,
                              dwpw=self.tcn_options['dwpw'],
                            )
        # -- initialize
        self._initialize_weights_randomly()
        self.optimizer = get_optimizer(config, self.parameters())
        self.lr_scheduler = CosineScheduler(config['lr'], config['epoch'])


    def forward(self, x, lengths):
        if self.modality == 'video':
            B, C, T, H, W = x.size()
            x = self.frontend3D(x)
            Tnew = x.shape[2]    # output should be B x C2 x Tnew x H x W
            x = threeD_to_2D_tensor( x )
            x = self.trunk(x)
            if self.backbone_type == 'shufflenet':
                x = x.view(-1, self.stage_out_channels)
            x = x.view(B, Tnew, x.size(1))
        elif self.modality == 'raw_audio':
            B, C, T = x.size()
            x = self.trunk(x)
            x = x.transpose(1, 2)
            lengths = [_//640 for _ in lengths]

        return x if self.extract_feats else self.tcn(x, lengths, B)

    def learn(self, data: dict, step: float, mode: str = 'train'):
        input = data['input']
        lengths = data['lengths']
        labels = data['labels']

        input, labels_a, labels_b, lam = mixup_data(input, labels, self.config['alpha'])
        labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

        self.zero_grad()

        logits = self.forward(input.unsqueeze(1).cuda(), lengths=lengths)
        loss_func = mixup_criterion(labels_a, labels_b, lam)
        loss = loss_func(self.criterion, logits)

        self.scalar_summaries['loss/{}/total'.format(mode)] \
            += [loss.detach().cpu().item()]

        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def evaluate(self, data: dict, step: int, mode: str):
        input = data['input']
        lengths = data['lengths']
        labels = data['labels']

        with torch.no_grad():
            logits = self.forward(input.unsqueeze(1).cuda(), lengths=lengths)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects = preds.eq(labels.cuda().view_as(preds)).sum().item()
            loss = self.criterion(logits, labels.cuda())
            running_loss = loss.item() * input.size(0)

        return running_corrects, running_loss

    def _initialize_weights_randomly(self):

        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))
