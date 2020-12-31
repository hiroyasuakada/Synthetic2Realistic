import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
import util.task as task
from util.GP import GPStruct
from .base_model import BaseModel
from . import network
from util.loss import ssim

class T2NetModel(BaseModel):
    def name(self):
        return 'T2Net model'

    def initialize(self, opt, labeled_dataset=None, unlabeled_dataset=None):
        BaseModel.initialize(self, opt)

        self.loss_names = ['img_rec', 'img_G', 'img_D', 'lab_s', 'lab_t', 'f_G', 'f_D', 'lab_smooth']
        self.visual_names = ['img_s', 'img_t', 'lab_s', 'lab_t', 'img_s2t', 'img_t2t', 'lab_s_g', 'lab_t_g']

        if self.isTrain:
            self.model_names = ['img2task', 's2t', 'img_D', 'f_D']
        else:
            self.model_names = ['img2task', 's2t']

        # define the transform network
        self.net_s2t = network.define_G(opt.image_nc, opt.image_nc, opt.ngf, opt.transform_layers, opt.norm,
                                                  opt.activation, opt.trans_model_type, opt.init_type, opt.drop_rate,
                                                  False, opt.gpu_ids, opt.U_weight)
        # define the task network
        self.net_img2task = network.define_G(opt.image_nc, opt.label_nc, opt.ngf, opt.task_layers, opt.norm,
                                            opt.activation, opt.task_model_type, opt.init_type, opt.drop_rate,
                                            False, opt.gpu_ids, opt.U_weight)

        # define the discriminator
        if self.isTrain:
            self.net_img_D = network.define_D(opt.image_nc, opt.ndf, opt.image_D_layers, opt.num_D, opt.norm,
                                              opt.activation, opt.init_type, opt.gpu_ids)
            self.net_f_D = network.define_featureD(opt.image_feature, opt.feature_D_layers, opt.norm,
                                                   opt.activation, opt.init_type, opt.gpu_ids)

        if self.isTrain:
            self.fake_img_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.l1loss = torch.nn.L1Loss()
            self.nonlinearity = torch.nn.ReLU()
            # initialize optimizers
            self.optimizer_T2Net = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.net_s2t.parameters())},
                                                     {'params': filter(lambda p: p.requires_grad, self.net_img2task.parameters()),
                                                      'lr': opt.lr_task, 'betas': (0.95, 0.999)}],
                                                    lr=opt.lr_trans, betas=(0.5, 0.9))
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_img_D.parameters()),
                                                                filter(lambda p: p.requires_grad, self.net_f_D.parameters())),
                                                lr=opt.lr_trans, betas=(0.5, 0.9))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_T2Net)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)


        # initializing GPstruct
        if self.isTrain and opt.gp:
            self.labeled_dataset = labeled_dataset
            self.unlabeled_dataset = unlabeled_dataset
            self.gp_struct = GPStruct(
                num_lbl=len(labeled_dataset), 
                num_unlbl=len(unlabeled_dataset),
                train_batch_size=self.opt.batch_size,
                version=self.opt.version,
                kernel_type=self.opt.kernel_type,
                pre_trained_enc = opt.pre_trained_enc,
                img_size = opt.load_size
            )


    def set_input(self, input):
        self.input = input
        self.img_source = input['img_source'].cuda(self.gpu_ids[0])
        self.img_target = input['img_target'].cuda(self.gpu_ids[0])
        if self.isTrain:
            self.lab_source = input['lab_source'].cuda(self.gpu_ids[0])
            self.lab_target = input['lab_target'].cuda(self.gpu_ids[0])

        # if len(self.gpu_ids) > 0:
        #     self.img_source = self.img_source.cuda(self.gpu_ids[0], async=True)
        #     self.img_target = self.img_target.cuda(self.gpu_ids[0], async=True)
        #     if self.isTrain:
        #         self.lab_source = self.lab_source.cuda(self.gpu_ids[0], async=True)
        #         self.lab_target = self.lab_target.cuda(self.gpu_ids[0], async=True)

    def forward(self):
        self.img_s = Variable(self.img_source)
        self.img_t = Variable(self.img_target)
        self.lab_s = Variable(self.lab_source)
        self.lab_t = Variable(self.lab_target)

    def backward_D_basic(self, netD, real, fake):

        D_loss = 0
        for (real_i, fake_i) in zip(real, fake):
            # Real
            D_real = netD(real_i.detach())
            # fake
            D_fake = netD(fake_i.detach())

            for (D_real_i, D_fake_i) in zip(D_real, D_fake):
                D_loss += (torch.mean((D_real_i-1.0)**2) + torch.mean((D_fake_i -0.0)**2))*0.5

        D_loss.backward()

        return D_loss

    def backward_D_image(self):
        network._freeze(self.net_s2t, self.net_img2task, self.net_f_D)
        network._unfreeze(self.net_img_D)
        size = len(self.img_s2t)
        fake = []
        for i in range(size):
            fake.append(self.fake_img_pool.query(self.img_s2t[i]))
        real = task.scale_pyramid(self.img_t, size)
        self.loss_img_D = self.backward_D_basic(self.net_img_D, real, fake)

    def backward_D_feature(self):
        network._freeze(self.net_s2t, self.net_img2task, self.net_img_D)
        network._unfreeze(self.net_f_D)
        self.loss_f_D = self.backward_D_basic(self.net_f_D, [self.lab_f_t], [self.lab_f_s])

    def foreward_G_basic(self, net_G, img_s, img_t):

        img = torch.cat([img_s, img_t], 0)
        fake = net_G(img)

        size = len(fake)

        f_s, f_t = fake[0].chunk(2)
        img_fake = fake[1:]

        img_s_fake = []
        img_t_fake = []

        for img_fake_i in img_fake:
            img_s, img_t = img_fake_i.chunk(2)
            img_s_fake.append(img_s)
            img_t_fake.append(img_t)

        return img_s_fake, img_t_fake, f_s, f_t, size

    def backward_synthesis2real(self):

        # image to image transform
        network._freeze(self.net_img2task, self.net_img_D, self.net_f_D)
        network._unfreeze(self.net_s2t)
        self.img_s2t, self.img_t2t, self.img_f_s, self.img_f_t, size = \
            self.foreward_G_basic(self.net_s2t, self.img_s, self.img_t)

        # image GAN loss and reconstruction loss
        img_real = task.scale_pyramid(self.img_t, size - 1)
        G_loss = 0
        rec_loss = 0
        for i in range(size - 1):
            rec_loss += self.l1loss(self.img_t2t[i], img_real[i])
            D_fake = self.net_img_D(self.img_s2t[i])
            for D_fake_i in D_fake:
                G_loss += torch.mean((D_fake_i - 1.0) ** 2)

        self.loss_img_G = G_loss * self.opt.lambda_gan_img
        self.loss_img_rec = rec_loss * self.opt.lambda_rec_img

        total_loss = self.loss_img_G + self.loss_img_rec

        total_loss.backward(retain_graph=True)

    def backward_translated2depth(self):

        # task network
        network._freeze(self.net_img_D, self.net_f_D)
        network._unfreeze(self.net_s2t, self.net_img2task)
        fake = self.net_img2task.forward(self.img_s2t[-1])

        size=len(fake)
        self.lab_f_s = fake[0]
        self.lab_s_g = fake[1:]

        #feature GAN loss
        D_fake = self.net_f_D(self.lab_f_s)
        G_loss = 0
        for D_fake_i in D_fake:
            G_loss += torch.mean((D_fake_i - 1.0) ** 2)
        self.loss_f_G = G_loss * self.opt.lambda_gan_feature

        # task loss
        lab_real = task.scale_pyramid(self.lab_s, size-1)
        task_loss = 0
        for (lab_fake_i, lab_real_i) in zip(self.lab_s_g, lab_real):
            task_loss += self.l1loss(lab_fake_i, lab_real_i)

        self.loss_lab_s = task_loss * self.opt.lambda_rec_lab

        # # loss ssim
        # if self.opt.lambda_task_ssim > 0:
        #     self.loss_ssim_s = torch.clamp((1 - ssim(self.lab_s_g[-1], self.lab_s, val_range = 1.0 )) * 0.5, 0, 1) * self.opt.lambda_task_ssim
        # else:
        #     self.loss_ssim_s = 0


        # total_loss = self.loss_f_G + self.loss_lab_s
        total_loss = self.loss_f_G + self.loss_lab_s

        total_loss.backward()

    def backward_real2depth(self):

        # image2depth
        network._freeze(self.net_s2t, self.net_img_D, self.net_f_D)
        network._unfreeze(self.net_img2task)
        fake = self.net_img2task.forward(self.img_t)
        size = len(fake)

        # Gan depth
        self.lab_f_t = fake[0]
        self.lab_t_g = fake[1:]

        img_real = task.scale_pyramid(self.img_t, size - 1)
        self.loss_lab_smooth = task.get_smooth_weight(self.lab_t_g, img_real, size-1) * self.opt.lambda_smooth

        total_loss = self.loss_lab_smooth

        total_loss.backward()

    def optimize_parameters(self, epoch_iter):

        self.forward()
        # T2Net
        self.optimizer_T2Net.zero_grad()
        self.backward_synthesis2real()
        self.backward_translated2depth()
        self.backward_real2depth()
        self.optimizer_T2Net.step()
        # Discriminator
        self.optimizer_D.zero_grad()
        self.backward_D_feature()
        self.backward_D_image()
        if epoch_iter % 5 == 0:
            self.optimizer_D.step()
            for p in self.net_f_D.parameters():
                p.data.clamp_(-0.01,0.01)

    def validation_target(self):

        lab_real = task.scale_pyramid(self.lab_t, len(self.lab_t_g))
        task_loss = 0
        for (lab_fake_i, lab_real_i) in zip(self.lab_t_g, lab_real):
            task_loss += task.rec_loss(lab_fake_i, lab_real_i)

        self.loss_lab_t = task_loss * self.opt.lambda_rec_lab

    def generate_fmaps_GP(self):
        self.gp_struct.gen_featmaps(self.labeled_dataset, self.net_img2task, self.device)
        self.gp_struct.gen_featmaps_unlbl(self.unlabeled_dataset, self.net_img2task, self.device)

    def optimize_parameters_GP(self, iter, data):
        input_im = data['img_target'].cuda(self.gpu_ids[0])
        # gt = data['lab_target'].cuda(self.device)
        imgid = data['img_target_paths']

        self.optimizer_T2Net.zero_grad()
        network._freeze(self.net_s2t, self.net_img_D, self.net_f_D)
        network._unfreeze(self.net_img2task)
        self.net_img2task.train()

        ### center in 
        # outputs = self.netTask(input_im)
        # zy_in = outputs[0]

        ### center_out
        _, zy_in = self.net_img2task(input_im, gp=True)

        loss_gp = self.gp_struct.compute_gploss(zy_in, imgid, iter, 0)
        self.loss_gp = loss_gp * self.opt.lambda_gp
        self.loss_gp.backward()
        self.optimizer_T2Net.step()  