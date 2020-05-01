import numpy as np
import torch
import utils.pc_utils as pc_utils

RegRec_SCALER = 8.0


def deform_input(X, lookup, RegRec_dist='gaussian', device='cuda:0'):
    """
    Deform a region in the point cloud. For more details see https://arxiv.org/pdf/2003.12641.pdf
    Input:
        args - commmand line arguments
        X - Point cloud [B, C, N]
        lookup - regions center point
        device - cuda/cpu
    Return:
        X - Point cloud with a deformed region
        mask - 0/1 label per point indicating if the point was centered
    """

    # get points' regions 
    regions = pc_utils.assign_region_to_point(X, device)

    n = pc_utils.NREGIONS
    region_ids = np.random.permutation(n ** 3)
    mask = torch.zeros_like(X).to(device)  # binary mask of deformed points

    for b in range(X.shape[0]):
        if RegRec_dist == 'point_collapse':
            X[b, :, :], indices = pc_utils.collapse_to_point(X[b, :, :], device)
            mask[b, :3, indices] = 1
        else:
            for i in region_ids:
                ind = regions[b, :] == i
                # if there are enough points in the region
                if torch.sum(ind) >= pc_utils.MIN_POINTS:
                    region = lookup[i].cpu().numpy()  # current region average point
                    mask[b, :3, ind] = 1
                    num_points = int(torch.sum(ind).cpu().numpy())
                    if RegRec_dist == 'gaussian':
                        rnd_pts = pc_utils.draw_from_gaussian(region, num_points)
                        X[b, :3, ind] = torch.tensor(rnd_pts, dtype=torch.float).to(device)
                    if RegRec_dist == 'uniform':
                        gap = 1 / 2 * (2 / n)  # gap between region center and region limit
                        rnd_pts = pc_utils.draw_from_uniform(gap, region, num_points)
                        X[b, :3, ind] = torch.tensor(rnd_pts, dtype=torch.float).to(device)
                    if RegRec_dist == 'gaussian_noise':
                        noise = 0.01 * torch.randn(num_points).to(device)
                        noise = noise.unsqueeze(0)
                        X[b, :3, ind] += noise
                    break  # move to the next shape in the batch
    return X, mask


def chamfer_distance(p1, p2, mask):
    """
    Calculate Chamfer Distance between two point sets
    Input:
        p1: size[B, C, N]
        p2: size[B, C, N]
    Return: 
        sum of all batches of Chamfer Distance of two point sets
    """

    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    # add dimension
    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    # repeat point values at the new dimension
    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    # calc norm between each point in p1 and each point in p2
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3) ** 2

    # add big value to points not in voxel and small 0 to those in voxel
    mask_cord = mask[:, :, 0]  # take only one coordinate  (batch_size, #points)
    m = mask_cord.clone()
    m[m == 0] = 100  # assign big value to points not in the voxel
    m[m == 1] = 0
    m = m.view(dist.size(0), 1, dist.size(2))  # transform to (batch_size, 1, #points)
    dist = dist + m

    # take the minimum distance for each point in p1 and sum over batch
    dist = torch.min(dist, dim=2)[0]  # for each point in p1 find the min in p2 (takes only from relevant ones because of the previous step)
    sum_pc = torch.sum(dist * mask_cord, dim=1)  # sum distances of each example (array broadcasting - zero distance of points not in the voxel for p1 and sum distances)
    dist = torch.sum(torch.div(sum_pc, torch.sum(mask_cord, dim=1)))  # divide each pc with the number of active points and sum
    return dist


def reconstruction_loss(pred, gold, mask):
    """
    Calculate symmetric chamfer Distance between predictions and labels
    Input:
        pred: size[B, C, N]
        gold: size[B, C, N]
        mask: size[B, C, N]
    Return: 
        mean batch loss
    """
    gold = gold.clone()

    batch_size = pred.size(0)

    # [batch_size, #points, coordinates]
    gold = gold.permute(0, 2, 1)
    mask = mask.permute(0, 2, 1)

    # calc average chamfer distance for each direction
    dist_gold = chamfer_distance(gold, pred, mask)
    dist_pred = chamfer_distance(pred, gold, mask)
    chamfer_loss = dist_gold + dist_pred

    # average loss
    loss = (1 / batch_size) * chamfer_loss

    return loss


def calc_loss(args, logits, labels, mask):
    """
    Calc. RegRec loss.
    Return: loss 
    """
    prediction = logits['RegRec']
    loss = args.RegRec_weight * reconstruction_loss(prediction, labels, mask) * RegRec_SCALER
    return loss
