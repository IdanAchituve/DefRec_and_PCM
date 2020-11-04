import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

K = 20

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, args, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    # Run on cpu or gpu
    if x.get_device() == -1 and args.gpus[0] != -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(x.get_device()) if args.gpus[0] != -1 else "cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous() # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]  # matrix [k*num_points*batch_size,3]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature


class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu', bias=True):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                    nn.ReLU(inplace=True)
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, activation='relu'):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                self.ac
             )

    def forward(self, x):
        x = self.fc(x)
        return x


class transform_net(nn.Module):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return: Transformation matrix of size 3xK """
    def __init__(self, args, in_ch, out=3):
        super(transform_net, self).__init__()
        self.K = out
        self.args = args

        self.conv2d1 = conv_2d(in_ch, 64, kernel=1, activation='leakyrelu', bias=False)
        self.conv2d2 = conv_2d(64, 128, kernel=1, activation='leakyrelu', bias=False)
        self.conv2d3 = conv_2d(128, 1024, kernel=1, activation='leakyrelu', bias=False)
        self.fc1 = fc_layer(1024, 512, activation='leakyrelu')
        self.fc2 = fc_layer(512, 256, activation='leakyrelu')
        self.fc3 = nn.Linear(256, out * out)

    def forward(self, x):
        # Run on cpu or gpu
        if x.get_device() == -1 and self.args.gpus[0] != -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(x.get_device()) if self.args.gpus[0] != -1 else "cpu")

        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = x.max(dim=-1, keepdim=False)[0]
        x = torch.unsqueeze(x, dim=3)
        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.K).view(1, self.K * self.K).repeat(x.size(0), 1)
        iden = iden.to(device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x


class shared_layers(nn.Module):
    def __init__(self, args, in_size=3):
        super(shared_layers, self).__init__()
        self.args = args
        self.k = K

        self.of1 = 64
        self.of2 = 64
        self.of3 = 64
        self.of4 = 64
        self.of5 = 64
        self.of6 = 1024

        self.conv1 = nn.Conv2d(in_size * 2, self.of1, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(self.of1, self.of2, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(self.of2 * 2, self.of3, kernel_size=1, bias=True)
        self.conv4 = nn.Conv2d(self.of3, self.of4, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(self.of4 * 2, self.of5, kernel_size=1, bias=True)
        num_f_prev = self.of1 + self.of3 + self.of5
        self.conv6 = nn.Conv1d(num_f_prev, self.of6, kernel_size=1, bias=True)


    def forward(self, x):

        batch_size = x.size(0)

        x = get_graph_feature(x, self.args, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, self.args, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, self.args, k=self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x123 = torch.cat((x1, x2, x3), dim=1)
        x4 = self.conv6(x123)

        x5 = F.adaptive_max_pool1d(x4, 1).view(batch_size, -1)

        return x123, x5.unsqueeze(2)

    def layers_sum(self):
        return self.of1 + self.of3 + self.of5


class DGCNN_DefRec(nn.Module):
    def __init__(self, args, in_size=3, num_classes=8):
        super(DGCNN_DefRec, self).__init__()

        self.args = args
        self.k = K

        self.input_transform_net = transform_net(args, in_size*2, in_size)
        self.shared_layers = shared_layers(args, in_size=in_size)

        self.num_f_prev = self.shared_layers.layers_sum()
        self.seg = segmentation(args, input_size=1024 + self.num_f_prev, num_classes=num_classes)
        self.DefRec = DeformationReconstruction(args, 1024 + self.num_f_prev, out_size=in_size)

    def forward(self, x, make_seg=True, activate_DefRec=True):

        num_points = x.size(2)
        logits = {}

        # Input transform net
        x0 = get_graph_feature(x, self.args, k=self.k)
        transformd_x0 = self.input_transform_net(x0)
        x = torch.matmul(transformd_x0, x)
        x123, x5 = self.shared_layers(x)

        x = torch.cat((x123, x5.repeat(1, 1, num_points)), dim=1)
        if make_seg:
            seg_logits = self.seg(x)
            logits["seg"] = seg_logits
        if activate_DefRec:
            rec_logits = self.DefRec(x)
            logits["DefRec"] = rec_logits

        return logits


class segmentation(nn.Module):
    def __init__(self, args, input_size, num_classes=8):
        super(segmentation, self).__init__()
        self.args = args
        self.of1 = 256
        self.of2 = 256
        self.of3 = 128

        self.bn1 = nn.BatchNorm1d(self.of1)
        self.bn2 = nn.BatchNorm1d(self.of2)
        self.bn3 = nn.BatchNorm1d(self.of3)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.dp2 = nn.Dropout(p=args.dropout)

        self.conv1 = nn.Conv1d(input_size, self.of1, kernel_size=1, bias=True)
        self.conv2 = nn.Conv1d(self.of1, self.of2, kernel_size=1, bias=True)
        self.conv3 = nn.Conv1d(self.of2, self.of3, kernel_size=1, bias=True)
        self.conv4 = nn.Conv1d(self.of3, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.dp1(F.relu(self.bn1(self.conv1(x))))
        x = self.dp2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x.permute(0, 2, 1)


class DeformationReconstruction(nn.Module):
    def __init__(self, args, input_size, out_size=3):
        super(DeformationReconstruction, self).__init__()
        self.args = args
        self.of1 = 256
        self.of2 = 256
        self.of3 = 128

        self.bn1 = nn.BatchNorm1d(self.of1)
        self.bn2 = nn.BatchNorm1d(self.of2)
        self.bn3 = nn.BatchNorm1d(self.of3)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.dp2 = nn.Dropout(p=args.dropout)

        self.conv1 = nn.Conv1d(input_size, self.of1, kernel_size=1, bias=True)
        self.conv2 = nn.Conv1d(self.of1, self.of2, kernel_size=1, bias=True)
        self.conv3 = nn.Conv1d(self.of2, self.of3, kernel_size=1, bias=True)
        self.conv4 = nn.Conv1d(self.of3, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.dp1(F.relu(self.bn1(self.conv1(x))))
        x = self.dp2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x.permute(0, 2, 1)