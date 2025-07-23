import os
import torch
import numpy as np
from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.function_utils import cycle


class Solver(object):
    def __init__(
        self,
        args,
        ae_model,
        inspector,
        model,
        data_loader,
        results_folder='./check_points'
    ):
        super(Solver, self).__init__()
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.ae_model = ae_model.to(self.device)
        self.inspector = inspector.to(self.device)
        self.train_epoch = args.train_epoch
        self.use_conv3d = args.use_conv3d

        self.data_loader = data_loader
        self.dl = cycle(data_loader)
        self.num_samples = len(data_loader.dataset)
        self.milestone_cycle = int(self.num_samples // args.batch_size)
        self.train_num_steps = args.train_epoch * self.milestone_cycle

        self.counter = 10
        self.step = 0
        self.mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
        self.mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
        self.hidden_size, self.link_size = model.size_2, model.size_1
        self.lambda_forward, self.lambda_backward, self.lambda_inv = args.lambda_forward, args.lambda_backward, args.lambda_inv
        self.lambda_mmd_forward = args.lambda_mmd_forward
        self.lambda_rec, self.lambda_gen = args.lambda_rec, args.lambda_gen
        self.lambda_gp = args.lambda_gp

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.ae_opt = Adam(filter(lambda p: p.requires_grad, self.ae_model.parameters()), lr=args.lr_ae)
        self.ins_opt = Adam(filter(lambda p: p.requires_grad, self.inspector.parameters()), lr=args.lr_ins)
        base = int(args.train_epoch // 10)

        self.sch = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[2*base, 7*base], gamma=0.25)

        if args.loss_type == "l1norm":
            self.criterion = nn.L1Loss().to(self.device)
        elif args.loss_type == "mse":
            self.criterion = nn.MSELoss().to(self.device)

        self.results_folder = Path(results_folder)
        os.makedirs(self.results_folder, exist_ok=True)

        self.known_rate = args.known_rate
        self.window = args.window
        self.batch_size = args.batch_size

    @staticmethod
    def MMD(x, y, widths_exponents, device):
        xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = torch.clamp(rx.t() + rx - 2.*xx, 0, np.inf)
        dyy = torch.clamp(ry.t() + ry - 2.*yy, 0, np.inf)
        dxy = torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))

        for C, a in widths_exponents:
            XX += C ** a * ((C + dxx) / a) ** -a
            YY += C ** a * ((C + dyy) / a) ** -a
            XY += C ** a * ((C + dxy) / a) ** -a

        return XX + YY - 2.*XY

    def loss_forward_mmd(self, y_hat, z_hat, y, z):
        yz_hat = torch.cat([y_hat, z_hat], dim=1)
        yz = torch.cat([y, z], dim=1)
        loss_mmd_zy = torch.mean(self.MMD(yz_hat, yz, self.mmd_forw_kernels, self.device))
        return self.lambda_mmd_forward * loss_mmd_zy

    def loss_reconstruction(self, x, y, a=1.):
        return a * self.criterion(x, y)

    def loss_generation(self, x_hat, a=1.):
        if self.use_conv3d:
            x_hat = x_hat.reshape(-1, self.window, x_hat.shape[-1])
        score = self.inspector(x_hat)
        return a * self.criterion(torch.ones_like(score), score)

    def loss_inspector(self, x, x_hat, unknown_index, a=1.):
        x_fake = x_hat.clone()
        x_fake[~unknown_index] = x[~unknown_index]
        if self.use_conv3d:
            x_fake = x_fake.reshape(-1, self.window, x.shape[-1])
        score = self.inspector(x_fake)
        standard = torch.zeros_like(score)
        standard[~unknown_index] = 1.
        return a * self.criterion(standard, score)

    def gradient_penalty(self, x, x_hat):
        alpha = torch.rand(x.shape[0], *[1] * (x.dim() - 1), device=self.device)
        interpolates = (alpha * x + (1 - alpha) * x_hat).requires_grad_(True)
        if self.use_conv3d:
            interpolates = interpolates.reshape(-1, self.window, x.shape[-1])
        inspector_interpolates = self.inspector(interpolates)

        grad_outputs = torch.ones_like(inspector_interpolates, device=self.device)
        gradients = torch.autograd.grad(outputs=inspector_interpolates, inputs=interpolates,
                                        grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return penalty

    def load(self):
        self.model.load_state_dict(torch.load(os.path.join(self.results_folder, 'Flow.pt')))
        self.ae_model.load_state_dict(torch.load(os.path.join(self.results_folder, 'AE.pt')))

    def train(self):
        counter_loss_gen = 0
        counter_loss_ins = 0
        self.ae_model.train()
        self.ae_model.requires_grad = True
        self.model.train()
        self.model.requires_grad = True
        self.inspector.train()
        self.inspector.requires_grad = True

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:
                x, y, unknown_index= next(self.dl)
                x, y, unknown_index = x.to(self.device), y.to(self.device), unknown_index.to(self.device)

                if self.use_conv3d:
                    b, w, d = x.shape
                    x = x.reshape(b * w, -1)
                    unknown_index = unknown_index.reshape(b * w, -1)
                    y = y.reshape(b * w, -1)
                b, d = x.shape[0], x.shape[-1]

                # Encode
                x_emb = self.ae_model.encode(x)

                # Flow forward
                y_hat, z_hat = self.model(x_emb)
                # Flow backward
                z = torch.randn(b, self.hidden_size - self.link_size).to(self.device)
                x_emb_hat = self.model(y, z, rev=True)
                x_emb_rec, jacobian = self.model(y_hat, z_hat, rev=True, cal_jacobian=True)

                # Decode
                x_hat = self.ae_model.decode(x_emb_hat)

                # Optimize Inspector
                self.ins_opt.zero_grad()
                gp = self.gradient_penalty(x, x_hat) if self.lambda_gp > 0 else 0
                loss_ins = self.loss_inspector(x, x_hat, unknown_index) + self.lambda_gp * gp
                loss_ins.backward(retain_graph=True)
                self.ins_opt.step()

                # Optimize Generator
                self.ae_opt.zero_grad()
                self.opt.zero_grad()
                loss_1 = self.loss_reconstruction(x_emb_rec, x_emb, self.lambda_inv)        # L_{inv}, Loss_1, lambda_1=1
                loss_2 = self.loss_reconstruction(y_hat, y, self.lambda_backward)           # L_{link}, Loss_2, lambda_2=1
                loss_3 = self.loss_forward_mmd(y_hat, z_hat, y, z)                          # L_{indep}, Loss_3, lambda_3=1
                # loss_4 = self.loss_reconstruction(x_emb_hat, x_emb, self.lambda_forward)  # L_{est}, Loss_4, lambda_4=2
                loss_5 = self.loss_reconstruction(x_hat[~unknown_index], x[~unknown_index], self.lambda_rec)
                loss_6 = self.loss_generation(x_hat, self.lambda_gen)
                loss_gen = loss_1 + loss_2 + loss_3 + loss_5 + loss_6
                loss_gen.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.ae_opt.step()
                self.opt.step()

                counter_loss_ins += loss_ins.item() / self.counter
                counter_loss_gen += loss_gen.item() / self.counter

                if self.step % self.counter == 0:
                    pbar.set_description(f'loss_ins: {counter_loss_ins:.6f}, loss_gen: {counter_loss_gen:.6f}')
                    counter_loss_ins = 0
                    counter_loss_gen = 0

                with torch.no_grad():
                    if self.step != 0 and self.step % self.milestone_cycle == 0:
                        self.sch.step()

                self.step += 1
                pbar.update(1)

        self.ae_model.eval()
        self.ae_model.requires_grad = False
        self.model.eval()
        self.model.requires_grad = False
        self.inspector.eval()
        self.inspector.requires_grad = False
        print(f'Training Complete.')
        torch.save(self.ae_model.state_dict(), os.path.join(self.results_folder, 'AE.pt'))
        torch.save(self.model.state_dict(), os.path.join(self.results_folder, 'Flow.pt'))
        torch.save(self.inspector.state_dict(), os.path.join(self.results_folder, 'Inspector.pt'))

    @torch.no_grad()
    def estimate(self, data_loader):
        self.model.eval()
        self.ae_model.eval()
        self.inspector.eval()

        estimations = np.empty([0, data_loader.dataset.dim_2])
        reals = np.empty([0, data_loader.dataset.dim_2])

        for idx, (x, y, _) in enumerate(data_loader):
            b,  _ = x.shape
            x, y = x.to(self.device), y.to(self.device)
            z = torch.randn(b, self.hidden_size - self.link_size).to(self.device)

            h_hat = self.model(y, z, rev=True)
            x_hat = self.ae_model.decode(h_hat)

            estimations = np.row_stack([estimations, x_hat.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])

        print(f'Estimating complete.')

        self.model.train()
        self.ae_model.train()
        self.inspector.train()

        return estimations, reals
