import torch


def mslossx(sim, sim_not, source, target, scale_pos=1.06, scale_neg=0.90, thresh=0.1, ms_mining=False, ms_margin=6.0):
    '''
    ref: http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    official codes: https://github.com/MalongTech/research-ms-loss
    '''
    adjacency = sim
    adjacency_not = sim_not

    mask_pos = adjacency.float()
    mask_neg = adjacency_not.float()

    sim_mat = torch.matmul(source, target.t()) / 2

    pos_pair = torch.mul(sim_mat, mask_pos)
    neg_pair = torch.mul(sim_mat, mask_neg)

    max_neg_pair = torch.max(torch.where(torch.abs(neg_pair) > 1e-6, neg_pair, -100 * torch.ones_like(neg_pair)),
                             dim=1, keepdim=True)[0]
    max_pos_pair = torch.max(torch.where(torch.abs(pos_pair) > 1e-6, pos_pair, -100 * torch.ones_like(pos_pair)),
                             dim=1, keepdim=True)[0]
    min_pos_pair = torch.min(torch.where(torch.abs(pos_pair) > 1e-6, pos_pair, 100 * torch.ones_like(pos_pair)),
                             dim=1, keepdim=True)[0]
    min_neg_pair = torch.min(torch.where(torch.abs(neg_pair) > 1e-6, neg_pair, 100 * torch.ones_like(neg_pair)),
                             dim=1, keepdim=True)[0]

    if ms_mining:
        margin = ms_margin
        margin_pos = margin_neg = margin
        mask_pos = torch.where(pos_pair - margin_pos < max_neg_pair, mask_pos, torch.zeros_like(mask_pos))
        mask_neg = torch.where(neg_pair + margin_neg > min_pos_pair, mask_neg, torch.zeros_like(mask_neg))

    pos_exp = torch.exp(-scale_pos * (pos_pair - thresh))
    pos_exp = torch.where(mask_pos > 0.0, pos_exp, torch.zeros_like(pos_exp))

    neg_exp = torch.exp(scale_neg * (neg_pair - thresh))
    neg_exp = torch.where(mask_neg > 0.0, neg_exp, torch.zeros_like(neg_exp))

    pos_term = torch.log(1.0 + pos_exp) / scale_pos
    neg_term = torch.log(1.0 + neg_exp) / scale_neg

    loss = torch.sum(pos_term + neg_term)
    return loss
