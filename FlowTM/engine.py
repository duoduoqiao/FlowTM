import os
import torch
import numpy as np

from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.function_utils import cycle, expectation_maximization


class Solver(object):
    def __init__(
        self,
        args,
        ae_model,
        model,
        data_loader,
        results_folder='./check_points'
    ):
        super(Solver, self).__init__()
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.ae_model = ae_model.to(self.device)
        self.train_epoch = args.train_epoch

        self.dl = cycle(data_loader)
        self.num_samples = len(data_loader.dataset)
        self.milestone_cycle = int(self.num_samples // args.batch_size)
        self.train_num_steps = args.train_epoch * self.milestone_cycle
        
        self.counter = 10
        self.step = 0
        self.train_lr = args.lr
        self.mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
        self.mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
        self.hidden_size, self.link_size = model.size_2, model.size_1
        self.lambda_forward, self.lambda_backward, self.lambda_inv = args.lambda_forward, args.lambda_backward, args.lambda_inv
        self.lambd_mmd_backward, self.lambd_mmd_forward = args.lambd_mmd_backward, args.lambd_mmd_forward 

        self.use_log_max = args.use_log_max
        if args.use_log_max:
            self.lambda_distribution = args.lambda_distribution 
        else:
            self.lambda_distribution = None
        
        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.ae_opt = Adam(filter(lambda p: p.requires_grad, self.ae_model.parameters()), lr=1e-3)
        base = int(args.train_epoch // 10)
        self.sch = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[7*base, 2*base], gamma=0.25)

        if args.loss_type == "l1norm":
            self.criteon = nn.L1Loss().to(self.device)
        elif args.loss_type == "mse":
            self.criteon = nn.MSELoss().to(self.device)

        self.results_folder = Path(results_folder)
        os.makedirs(self.results_folder, exist_ok=True)
        
    def loss_func(self, flow, link):
        b, _ = flow.shape
        z_sample = torch.randn(b, self.hidden_size - self.link_size).to(self.device)

        link_hat, z = self.model(flow)
        flow_rec, jacobian = self.model(link_hat, z, rev=True, cal_jacobian=True)
        flow_hat = self.model(link, z_sample, rev=True)
        loss_1 = self.loss_reconstruction(flow_hat, flow, self.lambda_forward)
        loss_2 = self.loss_reconstruction(link_hat, link, self.lambda_backward)
        loss_3 = self.loss_forward_mmd(link_hat, z, link, z_sample)
        loss_4 = self.loss_backward_mmd(flow_hat, flow)
        loss_5 = self.loss_reconstruction(flow_rec, flow, self.lambda_inv)

        if self.use_log_max:
            loss_6 = self.loss_max_likelihood(jacobian, link_hat, link, z)
            return loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6

        return loss_1 + loss_2 + loss_3 + loss_4 + loss_5
    
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
        return self.lambd_mmd_forward * loss_mmd_zy
    
    def loss_backward_mmd(self, x_hat, x):
        return self.lambd_mmd_backward * torch.mean(self.MMD(x_hat, x, self.mmd_back_kernels, self.device))

    def loss_max_likelihood(self, jac, y_hat, y, z):
        neg_log_likeli = torch.sum(z**2) + torch.sum((y_hat - y)**2) - jac
        return self.lambda_distribution * neg_log_likeli

    def loss_reconstruction(self, x, y, a=1.):
        return a * self.criteon(x, y)
    
    def load(self):
        self.model.load_state_dict(torch.load(os.path.join(self.results_folder, 'Flow.pt')))
        self.ae_model.load_state_dict(torch.load(os.path.join(self.results_folder, 'AE.pt')))

    def train(self):
        self.train_ae(self.train_epoch)
        device = self.device
        counter_loss = 0
        self.ae_model.eval()
        self.ae_model.requires_grad = False

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:
                x, y = next(self.dl)
                x, y = x.to(device), y.to(device)
                x = self.ae_model.encode(x)
                loss = self.loss_func(x, y)
                loss.backward()
                counter_loss += loss.item() / self.counter

                if self.step % self.counter == 0:
                    pbar.set_description(f'loss: {counter_loss:.6f}')
                    counter_loss = 0

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.milestone_cycle == 0:
                        self.sch.step()

                self.step += 1
                pbar.update(1)

        print('Training Complete')
        torch.save(self.model.state_dict(), os.path.join(self.results_folder, 'Flow.pt'))
        self.ae_model.train()
        self.ae_model.requires_grad = True
        self.train_d(int(self.train_epoch / 2))

    def train_ae(self, train_epochs):
        device = self.device
        counter_loss = 0

        with tqdm(initial=0, total=train_epochs * self.milestone_cycle) as pbar:

            while self.step < train_epochs * self.milestone_cycle:
                x, _ = next(self.dl)
                x = x.to(device)
                x_hat = self.ae_model(x)
                loss = self.criteon(x, x_hat)
                loss.backward()
                counter_loss += loss.item() / self.counter
                if self.step % self.counter == 0:
                    pbar.set_description(f'loss_ae: {counter_loss:.6f}')
                    counter_loss = 0
                
                self.ae_opt.step()
                self.ae_opt.zero_grad()
                self.step += 1
                pbar.update(1)
        
        self.step = 0
        print('Pre-training Complete')
        torch.save(self.ae_model.state_dict(), os.path.join(self.results_folder, 'AE.pt'))

    def train_d(self, train_epochs):
        self.step = 0
        device = self.device
        counter_loss = 0
        self.model.eval()
        self.model.requires_grad = False

        with tqdm(initial=0, total=train_epochs * self.milestone_cycle) as pbar:

            while self.step < train_epochs * self.milestone_cycle:
                x, y = next(self.dl)
                x, y = x.to(device), y.to(device)
                b, _ = x.shape
                z_sample = torch.randn(b, self.hidden_size - self.link_size).to(self.device)
                x_latent = self.model(y, z_sample, rev=True).detach()
                x_hat = self.ae_model.decode(x_latent)
                loss = self.criteon(x_hat, x)
                loss.backward()
                counter_loss += loss.item() / self.counter
                if self.step % self.counter == 0:
                    pbar.set_description(f'loss_ae: {counter_loss:.6f}')
                    counter_loss = 0
                
                self.ae_opt.step()
                self.ae_opt.zero_grad()
                self.step += 1
                pbar.update(1)
        
        self.step = 0
        self.model.train()
        self.model.requires_grad = True
        print('Post-training Complete')
        torch.save(self.ae_model.state_dict(), os.path.join(self.results_folder, 'AE.pt'))

    @torch.no_grad()
    def estimate(self, data_loader, rm=None):
        self.model.eval()
        self.ae_model.eval()
        
        # test_loss = []
        estimations = np.empty([0, data_loader.dataset.dim_2])
        reals = np.empty([0, data_loader.dataset.dim_2])

        for idx, (x, y) in enumerate(data_loader):
            b,  _ = x.shape
            x, y = x.to(self.device), y.to(self.device)
            z = torch.randn(b, self.hidden_size - self.link_size).to(self.device)

            h_hat = self.model(y, z, rev=True)
            x_hat = self.ae_model.decode(h_hat)
            if rm != None:
                x_hat = self.expectation_maximization(x_hat, y, rm, 5)
            estimations = np.row_stack([estimations, x_hat.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            # test_loss_y = self.criteon(x_hat, x)
            # test_loss.append(test_loss_y.item())

        # test_loss = np.average(test_loss)
        # print('Testing Mean Error:', test_loss.item())

        self.model.train()
        self.ae_model.train()
        return estimations, reals
