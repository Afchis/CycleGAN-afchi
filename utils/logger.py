import os
import numpy

import torch
import torch.nn.functional as F
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
                                                "loss_identity_G" : loss_identity_G}, self.epoch)

    def visual(self, img_A, img_B, same_A, same_B, fake_A, fake_B, res_A, res_B, iter):
        if self.vis == True and self.iter % iter == 0:
            out = torch.cat([img_A, img_B, fake_B, fake_A, res_A, res_B, same_A, same_B], dim=0).detach().cpu()
            grid_img = torchvision.utils.make_grid(out, nrow=self.batch_size*2)
            grid_img = torchvision.transforms.functional.to_pil_image(grid_img)
            grid_img.save("ignore/visual/" + self.dataset_name + "/%s/%i.png" % (self.tb, self.iter))

    def grad_norms(self, img_A, img_B, optimizer_G, optimizer_D_A, optimizer_D_B,
                   G_model_AtoB, G_model_BtoA, D_model_A, D_model_B, writer):
        if self.tb != "None" and self.disp["iter_epoch"] == 0:
            grad_norms = self._grad_norms_(img_A, img_B, optimizer_G, optimizer_D_A, optimizer_D_B,
                                           G_model_AtoB, G_model_BtoA, D_model_A, D_model_B)
            writer.add_scalars("%s_grad" % self.tb, {"loss_identity_AtoB" : grad_norms["loss_identity_AtoB"],
                                                     "loss_identity_BtoA" : grad_norms["loss_identity_BtoA"],
                                                     "loss_cycle_AtoB" : grad_norms["loss_cycle_AtoB"],
                                                     "loss_cycle_BtoA" : grad_norms["loss_cycle_BtoA"],
                                                     "loss_GAN_AtoB" : grad_norms["loss_GAN_AtoB"],
                                                     "loss_GAN_BtoA" : grad_norms["loss_GAN_BtoA"]}, (self.epoch-1))
            writer.add_scalars("%s_grad_first" % self.tb, {"loss_identity_AtoB_first_layer" : grad_norms["loss_identity_AtoB_first_layer"],
                                                           "loss_identity_BtoA_first_layer" : grad_norms["loss_identity_BtoA_first_layer"],
                                                           "loss_cycle_AtoB_first_layer" : grad_norms["loss_cycle_AtoB_first_layer"],
                                                           "loss_cycle_BtoA_first_layer" : grad_norms["loss_cycle_BtoA_first_layer"],
                                                           "loss_GAN_AtoB_first_layer" : grad_norms["loss_GAN_AtoB_first_layer"],
                                                           "loss_GAN_BtoA_first_layer" : grad_norms["loss_GAN_BtoA_first_layer"]}, (self.epoch-1))
            writer.add_scalars("%s_grad_last" % self.tb, {"loss_identity_AtoB_last_layer" : grad_norms["loss_identity_AtoB_last_layer"],
                                                          "loss_identity_BtoA_last_layer" : grad_norms["loss_identity_BtoA_last_layer"],
                                                          "loss_cycle_AtoB_last_layer" : grad_norms["loss_cycle_AtoB_last_layer"],
                                                          "loss_cycle_BtoA_last_layer" : grad_norms["loss_cycle_BtoA_last_layer"],
                                                          "loss_GAN_AtoB_last_layer" : grad_norms["loss_GAN_AtoB_last_layer"],
                                                          "loss_GAN_BtoA_last_layer" : grad_norms["loss_GAN_BtoA_last_layer"]}, (self.epoch-1))

    def _grad_norms_(self, img_A, img_B, optimizer_G, optimizer_D_A, optimizer_D_B,
                     G_model_AtoB, G_model_BtoA, D_model_A, D_model_B):
        grad_norms = dict()

        trues = torch.ones(img_A.size(0), 1).detach().to(img_A.device)
        lies = torch.zeros(img_A.size(0), 1).detach().to(img_A.device)

        def _l2_norm_grad_(model):      
            l2_norm_grad = torch.tensor([]).to(img_A.device)
            for name, param in model.named_parameters():
                if "weight" in name:
                    l2_norm_grad = torch.cat([l2_norm_grad, param.grad.view(-1)**2])
            return torch.sqrt(l2_norm_grad.sum()) 

        def _layer_mean_grad_(model):
            first_layer = torch.sqrt((model.state_dict()["model.1.weight"]**2).mean())
            last_layer = torch.sqrt((model.state_dict()["model.26.weight"]**2).mean())
            return first_layer, last_layer
        
        def _generator_():#img_A, img_B, G_model_AtoB, G_model_BtoA, D_model_A, D_model_B
            outs = dict()
            outs["same_B"] = G_model_AtoB(img_B)
            outs["same_A"] = G_model_BtoA(img_A)
            outs["fake_B"] = G_model_AtoB(img_A)
            outs["fake_A"] = G_model_BtoA(img_B)
            outs["pred_fake_A"] = D_model_A(outs["fake_A"])
            outs["pred_fake_B"] = D_model_B(outs["fake_B"])
            outs["res_B"] = G_model_AtoB(outs["fake_A"])
            outs["res_A"] = G_model_BtoA(outs["fake_B"])
            return outs

        outs = _generator_()

        optimizer_G.zero_grad()
        loss_identity_AtoB = F.l1_loss(outs["same_B"], img_B)
        loss_identity_AtoB.backward(retain_graph=True)
        grad_norms["loss_identity_AtoB"] = _l2_norm_grad_(G_model_AtoB)
        grad_norms["loss_identity_AtoB_first_layer"], grad_norms["loss_identity_AtoB_last_layer"] = _layer_mean_grad_(G_model_AtoB)

        optimizer_G.zero_grad()
        loss_identity_BtoA = F.l1_loss(outs["same_A"], img_A)
        loss_identity_BtoA.backward(retain_graph=True)
        grad_norms["loss_identity_BtoA"] = _l2_norm_grad_(G_model_BtoA)
        grad_norms["loss_identity_BtoA_first_layer"], grad_norms["loss_identity_BtoA_last_layer"] = _layer_mean_grad_(G_model_BtoA)

        optimizer_G.zero_grad()
        loss_cycle_AtoB = F.l1_loss(outs["res_A"], img_A)
        loss_cycle_AtoB.backward(retain_graph=True)
        grad_norms["loss_cycle_AtoB"] = _l2_norm_grad_(G_model_AtoB)
        grad_norms["loss_cycle_AtoB_first_layer"], grad_norms["loss_cycle_AtoB_last_layer"] = _layer_mean_grad_(G_model_AtoB)

        optimizer_G.zero_grad()
        loss_cycle_BtoA = F.l1_loss(outs["res_B"], img_B)
        loss_cycle_BtoA.backward(retain_graph=True)
        grad_norms["loss_cycle_BtoA"] = _l2_norm_grad_(G_model_BtoA)
        grad_norms["loss_cycle_BtoA_first_layer"], grad_norms["loss_cycle_BtoA_last_layer"] = _layer_mean_grad_(G_model_BtoA)

        optimizer_G.zero_grad()
        loss_GAN_AtoB = F.mse_loss(outs["pred_fake_B"], trues)
        loss_GAN_AtoB.backward(retain_graph=True)
        grad_norms["loss_GAN_AtoB"] = _l2_norm_grad_(G_model_AtoB)
        grad_norms["loss_GAN_AtoB_first_layer"], grad_norms["loss_GAN_AtoB_last_layer"] = _layer_mean_grad_(G_model_AtoB)

        optimizer_G.zero_grad()
        loss_GAN_BtoA = F.mse_loss(outs["pred_fake_A"], trues)
        loss_GAN_BtoA.backward(retain_graph=True)
        grad_norms["loss_GAN_BtoA"] = _l2_norm_grad_(G_model_BtoA)
        grad_norms["loss_GAN_BtoA_first_layer"], grad_norms["loss_GAN_BtoA_last_layer"] = _layer_mean_grad_(G_model_BtoA)
        return grad_norms