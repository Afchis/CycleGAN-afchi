import os
import itertools
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

# import class()
from model.generator import Generator
from model.discriminator import Discriminator
from utils.utils import LambdaLR
from utils.logger import Logger

# import def()
from dataloader.dataloader import Loader
from loss_metric.losses import TotalLoss, GenLoss, DisLoss
from utils.logger import PrintModelsParameters


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="horse2zebra", help="Dataset name")
parser.add_argument("--num_workers", type=int, default=12, help="Num worker")
parser.add_argument("--batch", type=int, default=6, help="Batch size")
parser.add_argument("--epochs", type=int, default=200, help="Num epochs")
parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
parser.add_argument("--vis", type=bool, default=False, help="Visual")
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument("--tb", type=str, default="None", help="Tensorboard name")

args = parser.parse_args()


# init tensorboard: !tensorboard --logdir=ignore/runs
writer = SummaryWriter('ignore/runs/%s' % args.tb)
print("Tensorboard name: %s" % args.tb)


# init data
print(" "*75, "\r", "Loading data...", end="\r")
train_loader = Loader(data_name=args.dataset, mode="train", batch_size=args.batch, num_workers=args.num_workers)
print("Dataloader len:", len(train_loader))


# init models
print(" "*75, "\r", "Loading models...", end="\r")
device = torch.device(args.device)
G_model_AtoB = Generator(3, 3).to(device)
G_model_BtoA = Generator(3, 3).to(device)
D_model_A = Discriminator(3).to(device)
D_model_B = Discriminator(3).to(device)
PrintModelsParameters(generator=G_model_AtoB, discriminator=D_model_A)

def save_model(model, dataset=args.dataset, tb=args.tb, mode="AtoB"):
    directory = "ignore/weigths/%s" % args.dataset
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Create directory: " + directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
    torch.save(model.state_dict(), "ignore/weights/%s/%s_s.pth" % (dataset, tb, mode))

# init optimizer and scheduler
print(" "*75, "\r", "Loading optimizer...", end="\r")
optimizer_G = torch.optim.Adam(itertools.chain(G_model_AtoB.parameters(), G_model_BtoA.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_model_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_model_B.parameters(), lr=args.lr, betas=(0.5, 0.999))
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.epochs, 0, 100).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.epochs, 0, 100).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.epochs, 0, 100).step)


def train(dataset=args.dataset, tb=args.tb):
    print(" "*75, "\r", "Loading training...", end="\r")
    logger = Logger(dataset=args.dataset, tb=args.tb, batch_size=args.batch,
                    vis=args.vis ,len_data=len(train_loader))
    for epoch in range(args.epochs):
        logger.init()
        for i, data in enumerate(train_loader):
            img_A, img_B = data
            img_A, img_B = img_A.to(device), img_B.to(device)
            logger.grad_norms(img_A, img_B, optimizer_G, optimizer_D_A, optimizer_D_B,
                              G_model_AtoB, G_model_BtoA, D_model_A, D_model_B, writer)
            optimizer_G.zero_grad()
            # Generators:
                 ## make same img for identity loss:
            same_B = G_model_AtoB(img_B)
            same_A = G_model_BtoA(img_A)
                 ## make fake img and use discrimitanor for GAN loss:
            fake_B = G_model_AtoB(img_A)
            fake_A = G_model_BtoA(img_B)
            pred_fake_A = D_model_A(fake_A)
            pred_fake_B = D_model_B(fake_B)
                 ## restore img for cycle loss:
            res_B = G_model_AtoB(fake_A)
            res_A = G_model_BtoA(fake_B)
            loss_GAN_G, loss_cycle_G, loss_identity_G = GenLoss(img_A, img_B, same_A, same_B, res_A, res_B, pred_fake_A, pred_fake_B)
            loss_G = loss_GAN_G + 10.*loss_cycle_G + 5.*loss_identity_G
            loss_G.backward()
            optimizer_G.step()
            # Discrimitators:
                ## D_A:
            optimizer_D_A.zero_grad()
            pred_fake_A_detach = D_model_A(fake_A.detach())
            pred_real_A = D_model_A(img_A)
            loss_D_A = DisLoss(pred_fake_A_detach, pred_real_A)
            loss_D_A.backward()
            optimizer_D_A.step()
                ## D_B:
            optimizer_D_B.zero_grad()
            pred_fake_B_detach = D_model_B(fake_B.detach())
            pred_real_B = D_model_B(img_B)
            loss_D_B = DisLoss(pred_fake_B_detach, pred_real_B)
            loss_D_B.backward()  
            optimizer_D_B.step()
            # logging:
            logger.update("iter_epoch", None)
            logger.update("loss_G", loss_G)
            logger.update("loss_GAN_G", loss_GAN_G)
            logger.update("loss_cycle_G", loss_cycle_G)
            logger.update("loss_identity_G", loss_identity_G)
            logger.update("loss_D_A", loss_D_A)
            logger.update("loss_D_B", loss_D_B)
            logger.printer(iter=1)
            logger.visual(img_A, img_B, same_A, same_B, fake_A, fake_B, res_A, res_B,
                          iter=50)
        save_model(G_model_AtoB, dataset=args.dataset, tb=args.tb, mode="AtoB")
        save_model(G_model_BtoA, dataset=args.dataset, tb=args.tb, mode="BtoA")
        logger.printer_epoch()
        logger.tensorboard_epoch(writer=writer)


if __name__ == "__main__":
    train()

