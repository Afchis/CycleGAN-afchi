import torch
import torch.nn.functional as F


def TotalLoss(img_A, img_B, same_A, same_B, res_A, res_B, pred_fake_A, pred_fake_B,
              pred_fake_A_detach, pred_fake_B_detach, pred_real_A, pred_real_B):
    trues = torch.ones_like(pred_fake_A).detach()
    lies = torch.zeros_like(pred_fake_A).detach()

    loss_identity_G = F.l1_loss(same_A, img_A) + F.l1_loss(same_B, img_B)
    loss_cycle_G = F.l1_loss(res_A, img_B) + F.l1_loss(res_B, img_B)
    loss_GAN_G = F.mse_loss(pred_fake_A, trues) + F.mse_loss(pred_fake_B, trues)
    loss_G = 5.*loss_identity_G + 10.*loss_cycle_G + loss_GAN_G

    loss_D_A = 0.5*F.mse_loss(pred_fake_A_detach, lies) + 0.5*F.mse_loss(pred_real_A, trues)
    loss_D_B = 0.5*F.mse_loss(pred_fake_B, lies) + 0.5*F.mse_loss(pred_real_B, trues)
    return loss_G, loss_D_A, loss_D_B


def GenLoss(img_A, img_B, same_A, same_B, res_A, res_B, pred_fake_A, pred_fake_B):
    trues = torch.ones_like(pred_fake_A).detach()
    lies = torch.zeros_like(pred_fake_A).detach()
    loss_identity_G = F.l1_loss(same_A, img_A) + F.l1_loss(same_B, img_B)
    loss_cycle_G = F.l1_loss(res_A, img_A) + F.l1_loss(res_B, img_B)
    loss_GAN_G = F.mse_loss(pred_fake_A, trues) + F.mse_loss(pred_fake_B, trues)
    return loss_GAN_G, loss_cycle_G, loss_identity_G


def DisLoss(pred_fake_detach, pred_real):
    trues = torch.ones_like(pred_fake_detach).detach()
    lies = torch.zeros_like(pred_fake_detach).detach()
    return 0.5*F.mse_loss(pred_fake_detach, lies) + 0.5*F.mse_loss(pred_real, trues)

