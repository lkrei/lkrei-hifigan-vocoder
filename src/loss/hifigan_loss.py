import torch
import torch.nn.functional as F


def discriminator_loss(disc_real_outputs, disc_fake_outputs):
    loss = 0.0
    for dr, df in zip(disc_real_outputs, disc_fake_outputs):
        loss += torch.mean((dr - 1.0) ** 2) + torch.mean(df ** 2)
    return loss


def generator_loss(disc_fake_outputs):
    loss = 0.0
    for df in disc_fake_outputs:
        loss += torch.mean((df - 1.0) ** 2)
    return loss


def feature_matching_loss(disc_real_features, disc_fake_features):
    loss = 0.0
    for real_feats, fake_feats in zip(disc_real_features, disc_fake_features):
        for rf, ff in zip(real_feats, fake_feats):
            loss += F.l1_loss(rf, ff)
    return loss
