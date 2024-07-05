import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data
from datetime import datetime

from auxi.loss import ChamferLoss
from auxi.model import Discriminator, Encoder, MLP_Generator, PointTrans_Generator, DualTrans_Generator
from auxi.utils import Logger, save_ply_batch
from auxi.dataset import ShapeNet


# ======================================================= Usage =======================================================
# CUDA_VISIBLE_DEVICES=0 python train.py --generator MLP --class_choice chair
# CUDA_VISIBLE_DEVICES=0 python train.py --generator PointTrans --class_choice chair
# CUDA_VISIBLE_DEVICES=0 python train.py --generator DualTrans --class_choice chair

# ================================================== Argument parsing =================================================
parser = argparse.ArgumentParser()
parser.add_argument('--patchDim', type=int, default=2, help='always use two dimension patches [default: 2]')
parser.add_argument('--generator', type=str, default="MLP", help='generator modules [MLP, PointTrans, DualTrans]')
parser.add_argument('--class_choice', type=str, default="chair", help='object categories in dataset')
parser.add_argument('--lrate', type=float, default=0.0005, help='learning rate')
parser.add_argument('--d_lrate', type=float, default=0.0005, help='learning rate of discriminator')
parser.add_argument('--e_lrate', type=float, default=0.0005, help='learning rate of encoder')
parser.add_argument('--g_lrate', type=float, default=0.0005, help='learning rate of generator')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--nepoch', type=int, default=300, help='number of training epochs [default: 200]')
parser.add_argument('--npoint', type=int, default=2048, help='number of points in object [default: 2048]')
parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
parser.add_argument('--npatch', type=int, default=8, help='number of patches for generation [default: 8]')
parser.add_argument('--nwords', type=int, default=16, help='used in DualTrans generator [default: 16]')
parser.add_argument('--npatch_point', type=int, default=256, help='number of points on each patch [default: 256]')
parser.add_argument('--ndis_boost', type=int, default=5, help='enhance discriminator training')
parser.add_argument('--lambda_gp', type=float, default=10.0, help='lambda for gradient penalty')
parser.add_argument('--nlatent', type=int, default=1024)
parser.add_argument('--z_dim', type=int, default=128, help='dimension of the shape code')
parser.add_argument('--cont_dim', type=int, default=128, help='dimension of the contrastive code')
parser.add_argument('--loadmodel', type=bool, default=False, help='load model')

opt = parser.parse_args()
assert opt.npatch*opt.npatch_point == opt.npoint

opt.training_id = "VAEGAN_%s_%s_%dpatch_%dpts_%dep" % (opt.generator, opt.class_choice, opt.npatch, opt.npoint, opt.nepoch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netD = Discriminator(opt).to(device)
netE = Encoder(opt).to(device)

generator_dict = {'MLP': MLP_Generator, 'PointTrans': PointTrans_Generator, 'DualTrans': DualTrans_Generator}
netG = generator_dict.get(opt.generator)
netG = netG(opt).to(device)

if opt.loadmodel:
    model_path = "./log/%s/%s" % (opt.training_id, 'saved_model.pth')
    checkpoint = torch.load(model_path)
    netD.load_state_dict(checkpoint['Discriminator'])
    netE.load_state_dict(checkpoint['Encoder'])
    netG.load_state_dict(checkpoint['Generator'])

    opt.start_epoch = checkpoint['Epoch']
    print('Model of epoch %d restored' % checkpoint['Epoch'])

dataset_train = ShapeNet(train='full', options=opt)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=4, drop_last=True)

shared_params = list(netD.conv1.parameters()) + \
                list(netD.conv2.parameters()) + \
                list(netD.conv3.parameters())
discri_params = list(netD.fc1.parameters()) + \
                list(netD.fc2.parameters()) + \
                list(netD.fc3.parameters())
contra_params = list(netD.cont_head1.parameters()) + \
                list(netD.cont_head2.parameters()) + \
                list(netD.cont_head3.parameters())


def lr_adaptive(epoch):
    if opt.generator in ['MLP', 'PointTrans']:
        if epoch < 100:
            return 1.0
        elif epoch < 200:
            return 0.2
        else:
            return 0.1
    else:
        if epoch < 200:
            return 0.2
        else:
            return 0.1


optimizer_shared = torch.optim.Adam(filter(lambda p: p.requires_grad, shared_params), opt.d_lrate, betas=(0.5, 0.999))
scheduler_shared = torch.optim.lr_scheduler.LambdaLR(optimizer_shared, lr_lambda=lr_adaptive)
optimizer_discri = torch.optim.Adam(filter(lambda p: p.requires_grad, discri_params), opt.d_lrate, betas=(0.5, 0.999))
scheduler_discri = torch.optim.lr_scheduler.LambdaLR(optimizer_discri, lr_lambda=lr_adaptive)
optimizer_contra = torch.optim.Adam(filter(lambda p: p.requires_grad, contra_params), opt.d_lrate, betas=(0.5, 0.999))
scheduler_contra = torch.optim.lr_scheduler.LambdaLR(optimizer_contra, lr_lambda=lr_adaptive)

optimizerE = torch.optim.Adam(netE.parameters(), lr=opt.e_lrate, betas=(0.5, 0.999))
schedulerE = torch.optim.lr_scheduler.LambdaLR(optimizerE, lr_lambda=lr_adaptive)
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.g_lrate, betas=(0.5, 0.999))
schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lr_adaptive)

opt.nbatch = len(dataset_train) % opt.batch_size
opt.nsyn_samples = 10 * opt.batch_size

opt.sample_saver_step = np.arange(10, opt.nepoch + 1, 5)
opt.model_saver_step = np.arange(20, opt.nepoch + 1, 20)
if opt.nepoch not in opt.sample_saver_step:
    opt.sample_saver_step = np.hstack([opt.sample_saver_step, np.array([opt.nepoch])])
if opt.nepoch not in opt.model_saver_step:
    opt.model_saver_step = np.hstack([opt.model_saver_step, np.array([opt.nepoch])])

logger_path = "./log/%s" % opt.training_id + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
if not os.path.exists(logger_path):
    os.makedirs(logger_path)

loss_d_log = Logger()
loss_e_log = Logger()
loss_g_log = Logger()
loss_c_log = Logger()


def compute_gradient_penalty(network, real_data, fake_data):
    alpha = torch.rand([opt.batch_size] + [1] * (real_data.dim() - 1))
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    d_interpolates = network(interpolates)[1]
    fake = torch.ones(d_interpolates.size(), dtype=torch.float32, requires_grad=False)
    fake = fake.to(device)

    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.contiguous()
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ====================================================== Training =====================================================
for epoch_id in range(opt.start_epoch, opt.start_epoch + opt.nepoch):
    start_time = time.time()

    netD.train()
    netE.train()
    netG.train()

    loss_d_log.reset()
    loss_e_log.reset()
    loss_g_log.reset()
    loss_c_log.reset()

    for batch_id, batch in enumerate(dataloader_train):
        batch = batch.to(device)
        label_real = torch.ones(opt.batch_size, 1).to(device)
        label_fake = torch.zeros(opt.batch_size, 1).to(device)

        #  Discriminator
        for _ in range(opt.ndis_boost):
            optimizer_shared.zero_grad()
            optimizer_discri.zero_grad()

            mu, logvar, z = netE(batch)
            points_fake = netG(z)[0]
            z_p = torch.randn(opt.batch_size, opt.z_dim).to(device)
            points_prior = netG(z_p)[0]

            _, logits_r = netD(batch)
            _, logits_f = netD(points_fake.detach())
            _, logits_p = netD(points_prior.detach())
            loss_d = 0.5*torch.mean(logits_f) + 0.5*torch.mean(logits_p) - torch.mean(logits_r)

            # gradient penalty
            gradient_penalty = compute_gradient_penalty(netD, batch, points_prior.detach())
            loss_d += opt.lambda_gp * gradient_penalty
            loss_d.backward()

            optimizer_shared.step()
            optimizer_discri.step()

            loss_d_log.add(loss_d.item())

        # Encoder and Generator
        optimizerE.zero_grad()
        optimizerG.zero_grad()

        mu, logvar, z = netE(batch)
        points_fake = netG(z)[0]
        z_p = torch.randn(opt.batch_size, opt.z_dim).to(device)
        points_prior = netG(z_p)[0]

        _, logits_f = netD(points_fake)
        _, logits_p = netD(points_prior)

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_kld = 0.25 * kld / opt.batch_size
        loss_recons = 0.25 * ChamferLoss(batch, points_fake, reduction='sum') / opt.batch_size
        loss_latent = 0.

        loss_e = loss_kld
        loss_e.backward(retain_graph=True)
        loss_g = loss_recons - 0.5*torch.mean(logits_f) - 0.5*torch.mean(logits_p)
        loss_g.backward()

        optimizerE.step()
        optimizerG.step()

        loss_e_log.add(loss_e.item())
        loss_g_log.add(loss_g.item())

        # Contrastive
        optimizerE.zero_grad()
        optimizerG.zero_grad()
        optimizer_shared.zero_grad()
        optimizer_contra.zero_grad()

        mu, logvar, z = netE(batch)
        points_fake = netG(z)[0]
        z_p = torch.randn(opt.batch_size, opt.z_dim).to(device)
        points_prior = netG(z_p)[0]

        with torch.no_grad():
            # for p, p_mom in zip(netD.parameters(), netM.parameters()):
            #     p_mom.data = (p_mom.data * 0.999) + (p.data * (1.0 - 0.999))
            # d_k = netM(batch, mode="cont")

            d_r = netD(batch, mode='cont')
            for layer in range(len(d_r)):
                d_r[layer] = F.normalize(d_r[layer], dim=1)

        d_f = netD(points_fake, mode="cont")
        d_p = netD(points_prior, mode='cont')
        for layer in range(len(d_f)):
            d_f[layer] = F.normalize(d_f[layer], dim=1)
            d_p[layer] = F.normalize(d_p[layer], dim=1)

        loss_c = torch.tensor(0.0).to(device)
        for layer in range(len(d_f)):
            l_pos = torch.einsum("nc,nc->n", d_f[layer], d_r[layer]).unsqueeze(-1)

            # fake points
            d_f_tensor = d_f[layer]

            # d_f_tensor, d_f_aug_tensor = d_f[layer], d_f_aug[layer]
            # l_pos = torch.einsum("nc,nc->n", d_f_tensor, d_f_aug_tensor).unsqueeze(-1)
            # l_pos = torch.einsum("nc,nc->n", d_f_tensor, d_f_tensor).unsqueeze(-1)
            # l_pos = torch.einsum("nc,nc->n", d_f_tensor, d_r_tensor).unsqueeze(-1)

            pos_tensor = torch.repeat_interleave(d_f_tensor, d_f_tensor.shape[0]-1, dim=0)
            neg_index = np.arange(d_f_tensor.shape[0]*d_f_tensor.shape[0])
            neg_index = neg_index[np.mod(neg_index, d_f_tensor.shape[0]+1) != 0]
            neg_tensor = d_f_tensor.repeat(d_f_tensor.shape[0], 1)
            neg_tensor = neg_tensor[neg_index]
            l_neg = torch.einsum("nc,nc->n", pos_tensor, neg_tensor).view(d_f_tensor.shape[0], -1)

            logits = torch.cat([l_pos, l_neg], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
            cont_loss_f = torch.nn.CrossEntropyLoss()(logits, labels)

            # prior points
            d_p_tensor = d_p[layer]

            # d_p_tensor, d_p_aug_tensor = d_p[layer], d_p_aug[layer]
            # l_pos = torch.einsum("nc,nc->n", d_p_tensor, d_p_aug_tensor).unsqueeze(-1)
            # l_pos = torch.einsum("nc,nc->n", d_p_tensor, d_p_tensor).unsqueeze(-1)
            # l_pos = torch.einsum("nc,nc->n", d_p_tensor, d_p_aug_tensor).unsqueeze(-1)

            pos_tensor = torch.repeat_interleave(d_p_tensor, d_p_tensor.shape[0]-1, dim=0)
            neg_index = np.arange(d_p_tensor.shape[0]*d_p_tensor.shape[0])
            neg_index = neg_index[np.mod(neg_index, d_p_tensor.shape[0]+1) != 0]
            neg_tensor = d_p_tensor.repeat(d_p_tensor.shape[0], 1)
            neg_tensor = neg_tensor[neg_index]
            l_neg = torch.einsum("nc,nc->n", pos_tensor, neg_tensor).view(d_p_tensor.shape[0], -1)

            logits = torch.cat([l_pos, l_neg], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
            cont_loss_p = torch.nn.CrossEntropyLoss()(logits, labels)

            loss_c = loss_c + 0.5*cont_loss_f + 0.5*cont_loss_p

        loss_c.backward()

        optimizerE.step()
        optimizerG.step()
        optimizer_shared.step()
        optimizer_contra.step()

        loss_c_log.add(loss_c.item())

    scheduler_shared.step()
    scheduler_discri.step()
    scheduler_contra.step()

    schedulerE.step()
    schedulerG.step()

    # Validation
    with torch.no_grad():
        netD.eval()
        netE.eval()
        netG.eval()

        if (epoch_id + 1) in opt.model_saver_step:
            torch.save({'Epoch': epoch_id + 1,
                        'Discriminator': netD.state_dict(),
                        'Encoder': netE.state_dict(),
                        'Generator': netG.state_dict()}, '%s/saved_model_%04d.pth' % (logger_path, (epoch_id + 1)))

        if (epoch_id + 1) in opt.sample_saver_step:
            syn_examples_path = os.path.join(logger_path, 'examples_%04d' % (epoch_id + 1))
            if not os.path.exists(syn_examples_path):
                os.mkdir(syn_examples_path)

            for j in range(int(opt.nsyn_samples / opt.batch_size)):
                noise = torch.randn(opt.batch_size, opt.z_dim).to(device)

                syn_examples, syn_examples_patches = netG(noise)
                syn_examples, syn_examples_patches = syn_examples.cpu().numpy(), syn_examples_patches.cpu().numpy()
                save_ply_batch(syn_examples, syn_examples_path, step=j)
                save_ply_batch(syn_examples_patches, syn_examples_path, step=j, patch=True)

    duration = time.time() - start_time
    print("Epoch: %04d" % (epoch_id + 1), 'time/m=%.4f' % (duration / 60.0),
          "loss_d=%.6f" % loss_d_log.mean(), "loss_e=%.6f" % loss_e_log.mean(),
          "loss_g=%.6f" % loss_g_log.mean(), "loss_c=%.6f" % loss_c_log.mean())
