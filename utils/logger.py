import os
import numpy
# import matplotlib.pyplot as plt

import torch
import torchvision


def PrintModelsParameters(generator, discriminator):
    Generator_parameters = filter(lambda p: p.requires_grad, generator.parameters())
    Discriminator_parameters = filter(lambda p: p.requires_grad, discriminator.parameters())
    params_G = sum([numpy.prod(p.size()) for p in Generator_parameters])
    params_D = sum([numpy.prod(p.size()) for p in Discriminator_parameters])
    print("Genetator parameters:", params_G)
    print("Discriminator parameters:", params_D)


class Logger():
    def __init__(self, tb, dataset, batch_size, vis, len_data):
        self.epoch = 0
        self.iter = 0
        self.tb = tb
        self.dataset_name = dataset
        self.batch_size = batch_size
        self.vis = vis
        self.len_data = len_data
        self.directory = "ignore/visual/%s/%s" % (self.dataset_name, self.tb)
        if self.vis == True:
            try:
                if not os.path.exists(self.directory):
                    os.makedirs(self.directory)
                    print("Create directory: " + self.directory)
            except OSError:
                print ('Error: Creating directory. ' +  self.directory)

    def init(self):
        self.epoch += 1
        self.disp = {
            "iter_epoch" : 0,
            "loss_G" : 0,
            "loss_GAN_G" : 0,
            "loss_cycle_G" : 0,
            "loss_identity_G" : 0,
            "loss_D_A" : 0,
            "loss_D_B" : 0
		}

    def update(self, key, x):
        if key == "iter_epoch":
            self.iter += 1
            self.disp[key] += 1
        else:
            self.disp[key] += x.detach().item()

    def printer(self, iter):
        if self.disp["iter_epoch"] % iter == 0:
            print(" "*75, end="\r")
            print(" Iter: %i (%0.2f%s) Losses: [G: %0.3f D_A: %0.3f D_B: %0.3f]"
                  % (self.iter, 100*self.disp["iter_epoch"]/self.len_data, chr(37),
                     self.disp["loss_G"]/self.disp["iter_epoch"],
                     self.disp["loss_D_A"]/self.disp["iter_epoch"],
                     self.disp["loss_D_B"]/self.disp["iter_epoch"]), end="\r")

    def printer_epoch(self):
        print(" "*75, end="\r")
        print("Epoch %i: loss_G: %0.4f loss_D_A: %0.4f loss_D_B: %0.4f"
              % (self.epoch, 
                 self.disp["loss_G"]/self.disp["iter_epoch"],
                 self.disp["loss_D_A"]/self.disp["iter_epoch"],
                 self.disp["loss_D_B"]/self.disp["iter_epoch"]))

    def tensorboard_epoch(self, writer):
        if self.tb != "None":
            loss_G = self.disp["loss_G"]/self.disp["iter_epoch"]
            loss_D_A = self.disp["loss_D_A"]/self.disp["iter_epoch"]
            loss_D_B = self.disp["loss_D_B"]/self.disp["iter_epoch"]
            loss_GAN_G = self.disp["loss_GAN_G"]/self.disp["iter_epoch"]
            loss_cycle_G = self.disp["loss_cycle_G"]/self.disp["iter_epoch"]
            loss_identity_G = self.disp["loss_identity_G"]/self.disp["iter_epoch"]
            writer.add_scalars("%s" % self.tb, {"loss_G" : loss_G,
                                                "loss_D_A" : loss_D_A,
                                                "loss_D_B" : loss_D_B,
                                                "loss_GAN_G" : loss_GAN_G,
                                                "loss_cycle_G" : loss_cycle_G,
                                                "loss_identity_G" : loss_identity_G,}, self.epoch)

    def visual(self, img_A, img_B, same_A, same_B, fake_A, fake_B, res_A, res_B,
               iter):
        if self.vis == True and self.iter % iter == 0:
            out = torch.cat([img_A, img_B, fake_B, fake_A, res_A, res_B, same_A, same_B], dim=0).detach().cpu()
            grid_img = torchvision.utils.make_grid(out, nrow=self.batch_size*2)
            grid_img = torchvision.transforms.functional.to_pil_image(grid_img)
            grid_img.save("ignore/visual/" + self.dataset_name + "/%s/%i.png" % (self.tb, self.iter))
            # if visual == True and self.disp["iter_epoch"] % 25 == 0:
            #     plt.imshow(grid_img.permute(1, 2, 0))
            #     plt.show(block=False)
            #     plt.pause(0.05)
            