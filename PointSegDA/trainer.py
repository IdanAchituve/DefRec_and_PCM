import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import argparse
import copy
import utils.log
from torchsummary import summary
from PointSegDA.data.dataloader import datareader
from PointSegDA.Models import DGCNN_DefRec
from utils import pc_utils
from sklearn.metrics import jaccard_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from DefRec_and_PCM import DefRec, PCM

NWORKERS=4
MAX_LOSS = 9 * (10**9)

def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
parser.add_argument('--exp_name', type=str, default='DefRec_PCM',  help='Name of the experiment')
parser.add_argument('--dataroot', type=str, default='./data/PointSegDAdataset', help='data path')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--src_dataset', type=str, default='adobe', choices=['adobe', 'faust', 'mit', 'scape'])
parser.add_argument('--trgt_dataset', type=str, default='faust', choices=['adobe', 'faust', 'mit', 'scape'])
parser.add_argument('--epochs', type=int, default=200, help='number of episode to train')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size', help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--DefRec_dist', type=str, default='volume_based_radius', metavar='N',
                    choices=['volume_based_voxels', 'volume_based_radius'],
                    help='distortion of points')
parser.add_argument('--radius', type=float, default=0.3, help='radius of the ball for reconstruction')
parser.add_argument('--min_pts', type=int, default=20, help='minimum number of points per region')
parser.add_argument('--num_regions', type=int, default=3, help='number of regions to split shape by')
parser.add_argument('--noise_std', type=float, default=0.1, help='learning rate')
parser.add_argument('--apply_PCM', type=str2bool, default=True, help='Using mixup in source')
parser.add_argument('--DefRec_weight', type=float, default=0.5, help='weight of the DefRec loss')
parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')

args = parser.parse_args()
# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))

random.seed(1)
np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')

# ==================
# Read Data
# ==================
src_trainset = datareader(args.dataroot, dataset=args.src_dataset, partition='train', domain='source')
src_valset = datareader(args.dataroot, dataset=args.src_dataset, partition='val', domain='source')

trgt_trainset = datareader(args.dataroot, dataset=args.trgt_dataset, partition='train', domain='target')
trgt_valset = datareader(args.dataroot, dataset=args.trgt_dataset, partition='val', domain='target')
trgt_testset = datareader(args.dataroot, dataset=args.trgt_dataset, partition='test', domain='target')

# dataloaders for source and target
batch_size = min(len(src_trainset), len(trgt_trainset), args.batch_size)
src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=batch_size,
                               shuffle=True, drop_last=True)
src_val_loader = DataLoader(src_valset, num_workers=NWORKERS, batch_size=args.test_batch_size)
trgt_train_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=batch_size,
                               shuffle=True, drop_last=True)
trgt_val_loader = DataLoader(trgt_valset, num_workers=NWORKERS, batch_size=args.test_batch_size)
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)

# ==================
# Init Model
# ==================
num_classes = 8
model = DGCNN_DefRec(args, in_size=3, num_classes=num_classes)

summary(model, input_size=(3, 2048), device='cpu')

model = model.to(device)

# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
best_model = copy.deepcopy(model)


# ==================
# Optimizer
# ==================
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd) \
    if args.optimizer == "SGD" \
    else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
t_max = args.epochs
scheduler = CosineAnnealingLR(opt, T_max=t_max, eta_min=0.0)

# ==================
# Loss and Metrics
# ==================
criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch
sample_criterion = nn.CrossEntropyLoss(reduction='none')  # to get the loss per shape

def seg_metrics(labels, preds):
    batch_size = labels.shape[0]
    mIOU = accuracy = 0
    for b in range(batch_size):
        y_true = labels[b, :].detach().cpu().numpy()
        y_pred = preds[b, :].detach().cpu().numpy()
        # IOU per class and average
        mIOU += jaccard_score(y_true, y_pred, average='macro')
        accuracy += np.mean(y_true == y_pred)
    return mIOU, accuracy


# ==================
# Validation/test
# ==================
def test(test_loader):

    # Run on cpu or gpu
    seg_loss = mIOU = accuracy = 0.0
    batch_idx = num_samples = 0

    with torch.no_grad():
        model.eval()
        for i, data in enumerate(test_loader):
            data, labels = data[0].to(device), data[1].to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.shape[0]

            logits = model(data, make_seg=True, activate_DefRec=False)
            loss = criterion(logits["seg"].permute(0, 2, 1), labels)
            seg_loss += loss.item() * batch_size

            # evaluation metrics
            preds = logits["seg"].max(dim=2)[1]
            batch_mIOU, batch_seg_acc = seg_metrics(labels, preds)
            mIOU += batch_mIOU
            accuracy += batch_seg_acc

            num_samples += batch_size
            batch_idx += 1

    seg_loss /= num_samples
    mIOU /= num_samples
    accuracy /= num_samples
    model.train()

    return seg_loss, mIOU, accuracy


# ==================
# Train
# ==================
src_best_val_acc = trgt_best_val_acc = best_val_epoch = 0
src_best_val_mIOU = trgt_best_val_mIOU = 0.0
src_best_val_loss = trgt_best_val_loss = MAX_LOSS
epoch = step = 0
lookup = torch.Tensor(pc_utils.region_mean(args.num_regions)).to(device)

for epoch in range(args.epochs):
    model.train()

    # init data structures for saving epoch stats
    # Run on cpu or gpu
    src_seg_loss = src_mIOU = src_accuracy = 0.0
    trgt_rec_loss = total_loss = 0.0
    batch_idx = src_count = trgt_count = 0

    for k, data in enumerate(zip(src_train_loader, trgt_train_loader)):
        step += 1
        opt.zero_grad()
        trgt_batch_loss = src_batch_loss = batch_mIOU = batch_seg_acc = 0.0

        #### source data ####
        if data[0] is not None:
            src_data, src_labels = data[0][0].to(device), data[0][1].to(device)

            # change to [batch_size, num_coordinates, num_points]
            src_data = src_data.permute(0, 2, 1)
            batch_size = src_data.shape[0]

            if args.apply_PCM:
                src_data, src_labels = PCM.mix_shapes_segmentation(args, src_data, src_labels)

            logits = model(src_data, make_seg=True, activate_DefRec=False)
            loss = (1 - args.DefRec_weight) * criterion(logits['seg'].permute(0, 2, 1), src_labels)
            loss.backward()

            src_batch_loss = loss.item()
            src_seg_loss += loss.item() * batch_size
            total_loss += loss.item() * batch_size

            # evaluation metrics
            preds = logits['seg'].max(dim=2)[1]

            batch_mIOU, batch_seg_acc = seg_metrics(src_labels, preds)
            src_mIOU += batch_mIOU
            src_accuracy += batch_seg_acc
            src_count += batch_size

        #### target data ####
        if data[1] is not None:
            trgt_data, trgt_labels = data[1][0].to(device), data[1][1].to(device)
            trgt_data = trgt_data.permute(0, 2, 1)
            batch_size = trgt_data.shape[0]
            trgt_data_orig = trgt_data.clone()

            trgt_data, trgt_mask = DefRec.deform_input(trgt_data, lookup, args.DefRec_dist, device=device)
            logits = model(trgt_data, make_seg=False, activate_DefRec=True)
            loss = DefRec.calc_loss(args, logits, trgt_data_orig, trgt_mask)

            trgt_batch_loss = loss.item()
            trgt_rec_loss += loss.item() * batch_size
            total_loss += loss.item() * batch_size
            loss.backward()

            trgt_count += batch_size

        batch_idx += 1
        opt.step()

    scheduler.step(epoch=epoch)

    # print progress
    trgt_rec_loss /= trgt_count
    src_seg_loss /= src_count
    src_mIOU /= src_count
    src_accuracy /= src_count

    #===================
    # Validation
    #===================
    src_val_loss, src_val_miou, src_val_acc = test(src_val_loader)
    trgt_val_loss, trgt_val_miou, trgt_val_acc = test(trgt_val_loader)

    # save model according to best source model (since we don't have target labels)
    if src_val_loss < src_best_val_loss:
        src_best_val_mIOU = src_val_miou
        src_best_val_acc = src_val_acc
        src_best_val_loss = src_val_loss
        trgt_best_val_mIOU = trgt_val_miou
        trgt_best_val_acc = trgt_val_acc
        trgt_best_val_loss = trgt_val_loss
        best_val_epoch = epoch
        best_model = copy.deepcopy(model)

    io.cprint(f"Epoch: {epoch}, "
              f"Target train rec loss: {trgt_rec_loss:.5f}, "
              f"Source train seg loss: {src_seg_loss:.5f}, "
              f"Source train seg mIOU: {src_mIOU:.5f}, "
              f"Source train seg accuracy: {src_accuracy:.5f}")

    io.cprint(f"Epoch: {epoch}, "
              f"Source val seg loss: {src_val_loss:.5f}, "
              f"Source val seg mIOU: {src_val_miou:.5f}, "
              f"Source val seg accuracy: {src_val_acc:.5f}")

    io.cprint(f"Epoch: {epoch}, "
              f"Target val seg loss: {trgt_val_loss:.5f}, "
              f"Target val seg mIOU: {trgt_val_miou:.5f}, "
              f"Target val seg accuracy: {trgt_val_acc:.5f}")

io.cprint("Best model was found at epoch %d\n"
          "source val seg loss: %.4f, source val seg mIOU: %.4f, source val seg accuracy: %.4f\n"
          "target val seg loss: %.4f, target val seg mIOU: %.4f, target val seg accuracy: %.4f\n"
         % (best_val_epoch,
            src_best_val_loss, src_best_val_mIOU, src_best_val_acc,
            trgt_best_val_loss, trgt_best_val_mIOU, trgt_best_val_acc))

#===================
# Test
#===================
model = best_model
trgt_test_loss, trgt_test_miou, trgt_test_acc = test(trgt_test_loader)
io.cprint("target test seg loss: %.4f, target test seg mIOU: %.4f, target test seg accuracy: %.4f"
          % (trgt_test_loss, trgt_test_miou, trgt_test_acc))