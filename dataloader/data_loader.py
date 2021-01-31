import os, glob
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from natsort import natsorted
from dataloader.image_folder import make_dataset


def dataloader(opt, gp=False):

    ######################################################################################################################
    if opt.isTrain and opt.mixed:
        datasets = CreateMixedDataset(opt)
        dataset = torch.utils.data.DataLoader(
            datasets, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=int(opt.num_threads), drop_last=True
        )
        return dataset

    if opt.taskmodel:
        datasets = CreateTaskDataset(opt)
        dataset = torch.utils.data.DataLoader(
            datasets, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=int(opt.num_threads), drop_last=True
        )
        return dataset    

    ######################################################################################################################


    if not gp:
        datasets = CreateFullDataset(opt)
        dataset = torch.utils.data.DataLoader(
            datasets, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=int(opt.num_threads), drop_last=True
        )
        return dataset
    else:
        labeled_datasets = CreateLabeledDataset(opt)
        unlabeled_datasets = CreateUnlabeledDataset(opt)
        labeled_dataset = torch.utils.data.DataLoader(
            labeled_datasets, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=int(opt.num_threads)
        )
        unlabeled_dataset = torch.utils.data.DataLoader(
            unlabeled_datasets, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=int(opt.num_threads)
        )
        return labeled_dataset, unlabeled_dataset
    

class CreateFullDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(CreateFullDataset, self).__init__()
        self.opt = opt

        self.img_source_paths, self.img_source_size = make_dataset(opt, opt.img_source_dir)  # img_source_dir = '^/trainA'
        self.img_target_paths, self.img_target_size = make_dataset(opt, opt.img_target_dir)  # img_source_dir = '^/trainB'

        self.lab_source_paths, self.lab_source_size = make_dataset(opt, opt.lab_source_dir)  # img_source_dir = '^/trainA_depth'
        # for visual results, not for training
        self.lab_target_paths, self.lab_target_size = make_dataset(opt, opt.lab_target_dir)  # img_source_dir = '^/trainB_depth'     
        
        if self.opt.isTrain:
            self.lab_source_paths, self.lab_source_size = make_dataset(opt, opt.lab_source_dir)  # img_source_dir = '^/trainA_depth'
            # for visual results, not for training
            self.lab_target_paths, self.lab_target_size = make_dataset(opt, opt.lab_target_dir)  # img_source_dir = '^/trainB_depth'

        self.transform_augment = self._make_transform(opt, True)
        self.transform_no_augment = self._make_transform(opt, False)  # for label

        self.transform_no_augment_depth = self._make_transform(opt, False, True)  # for label

    # get tensor data
    def __getitem__(self, item):
        index = random.randint(0, self.img_target_size - 1)
        img_source_path = self.img_source_paths[item % self.img_source_size]
        if self.opt.dataset_mode == 'paired':
            img_target_path = self.img_target_paths[item % self.img_target_size]
        elif self.opt.dataset_mode == 'unpaired':
            img_target_path = self.img_target_paths[index]
        else:
            raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

        img_source = Image.open(img_source_path).convert('RGB')
        img_target = Image.open(img_target_path).convert('RGB')
        img_source = img_source.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)
        img_target = img_target.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

        if self.opt.isTrain:
            lab_source_path = self.lab_source_paths[item % self.lab_source_size]
            if self.opt.dataset_mode == 'paired':
                lab_target_path = self.lab_target_paths[item % self.img_target_size]
            elif self.opt.dataset_mode == 'unpaired':
                lab_target_path = self.lab_target_paths[index]
            else:
                raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

            lab_source = Image.open(lab_source_path) #.convert('RGB')
            lab_target = Image.open(lab_target_path) #.convert('RGB')
            lab_source = lab_source.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)
            lab_target = lab_target.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

            img_source, lab_source, scale = self.paired_transform(self.opt, img_source, lab_source)
            img_source = self.transform_augment(img_source)
            lab_source = self.transform_no_augment_depth(lab_source)

            img_target, lab_target, scale = self.paired_transform(self.opt, img_target, lab_target)
            img_target = self.transform_no_augment(img_target)
            lab_target = self.transform_no_augment_depth(lab_target)

            return {'img_source': img_source, 'img_target': img_target,
                    'lab_source': lab_source, 'lab_target': lab_target,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    'lab_source_paths': lab_source_path, 'lab_target_paths': lab_target_path
                    }

        else:
            img_source = self.transform_augment(img_source)
            img_target = self.transform_no_augment(img_target)

            # return {'img_source': img_source, 'img_target': img_target,
            #         'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
            #         }


            ##############################################################################################
            lab_source_path = self.lab_source_paths[item % self.lab_source_size]
            if self.opt.dataset_mode == 'paired':
                lab_target_path = self.lab_target_paths[item % self.img_target_size]
            elif self.opt.dataset_mode == 'unpaired':
                lab_target_path = self.lab_target_paths[index]
            else:
                raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)
            lab_source = Image.open(lab_source_path) #.convert('RGB')
            lab_target = Image.open(lab_target_path) #.convert('RGB')
            lab_source = lab_source.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)
            lab_target = lab_target.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)
            lab_source = self.transform_no_augment_depth(lab_source)
            lab_target = self.transform_no_augment_depth(lab_target)

            return {'img_source': img_source, 'img_target': img_target,
                    'lab_source': lab_source, 'lab_target': lab_target,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    'lab_source_paths': lab_source_path, 'lab_target_paths': lab_target_path
                    }
            ##############################################################################################

    def __len__(self):
        return max(self.img_source_size, self.img_target_size)
        # return self.img_source_size

    def _make_transform(self, opt, augment, depth=False):
        transforms_list = []

        if augment:
            if opt.isTrain:
                transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
        
        transforms_list.append(transforms.ToTensor())
        
        if depth:
            transforms_list.append(transforms.Normalize((0.5, ), (0.5, )))
        else:
            transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(transforms_list)

    def paired_transform(self, opt, image, depth):
        scale_rate = 1.0

        if opt.flip:
            n_flip = random.random()
            if n_flip > 0.5:
                image = F.hflip(image)
                depth = F.hflip(depth)

        if opt.rotation:
            n_rotation = random.random()
            if n_rotation > 0.5:
                degree = random.randrange(-500, 500)/100
                image = F.rotate(image, degree, Image.BICUBIC)
                depth = F.rotate(depth, degree, Image.BILINEAR)

        return image, depth, scale_rate


class CreateLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(CreateLabeledDataset, self).__init__()
        self.opt = opt

        self.img_source_paths, self.img_source_size = make_dataset(opt, opt.img_source_dir)  # img_source_dir = '^/trainA'
        self.lab_source_paths, self.lab_source_size = make_dataset(opt, opt.lab_source_dir)  # img_source_dir = '^/trainA_depth'  

        self.transform_augment = self._make_transform(opt, True)
        self.transform_no_augment = self._make_transform(opt, False)  # for label

        self.transform_no_augment_depth = self._make_transform(opt, False, True)  # for label

    # get tensor data
    def __getitem__(self, item):
        img_source_path = self.img_source_paths[item % self.img_source_size]

        img_source = Image.open(img_source_path).convert('RGB')
        img_source = img_source.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

        lab_source_path = self.lab_source_paths[item % self.lab_source_size]

        lab_source = Image.open(lab_source_path) #.convert('RGB')
        lab_source = lab_source.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

        img_source, lab_source, scale = self.paired_transform(self.opt, img_source, lab_source)
        img_source = self.transform_augment(img_source)
        lab_source = self.transform_no_augment_depth(lab_source)

        return {'img_source': img_source, 'lab_source': lab_source, 
                'img_source_paths': img_source_path, 'lab_source_paths': lab_source_path
                }

    def __len__(self):
        return self.img_source_size

    def _make_transform(self, opt, augment, depth=False):
        transforms_list = []

        if augment:
            if opt.isTrain:
                transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
        
        transforms_list.append(transforms.ToTensor())
        
        if depth:
            transforms_list.append(transforms.Normalize((0.5, ), (0.5, )))
        else:
            transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(transforms_list)

    def paired_transform(self, opt, image, depth):
        scale_rate = 1.0

        if opt.flip:
            n_flip = random.random()
            if n_flip > 0.5:
                image = F.hflip(image)
                depth = F.hflip(depth)

        if opt.rotation:
            n_rotation = random.random()
            if n_rotation > 0.5:
                degree = random.randrange(-500, 500)/100
                image = F.rotate(image, degree, Image.BICUBIC)
                depth = F.rotate(depth, degree, Image.BILINEAR)

        return image, depth, scale_rate


class CreateUnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(CreateUnlabeledDataset, self).__init__()
        self.opt = opt

        self.img_target_paths, self.img_target_size = make_dataset(opt, opt.img_target_dir)  # img_source_dir = '^/trainB'
        self.lab_target_paths, self.lab_target_size = make_dataset(opt, opt.lab_target_dir)  # img_source_dir = '^/trainB_depth'     
        
        self.transform_augment = self._make_transform(opt, True)
        self.transform_no_augment = self._make_transform(opt, False)  # for label

        self.transform_no_augment_depth = self._make_transform(opt, False, True)  # for label

    # get tensor data
    def __getitem__(self, item):
        index = random.randint(0, self.img_target_size - 1)
        if self.opt.dataset_mode == 'paired':
            img_target_path = self.img_target_paths[item % self.img_target_size]
        elif self.opt.dataset_mode == 'unpaired':
            img_target_path = self.img_target_paths[index]
        else:
            raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

        img_target = Image.open(img_target_path).convert('RGB')
        img_target = img_target.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

        if self.opt.dataset_mode == 'paired':
            lab_target_path = self.lab_target_paths[item % self.img_target_size]
        elif self.opt.dataset_mode == 'unpaired':
            lab_target_path = self.lab_target_paths[index]
        else:
            raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

        lab_target = Image.open(lab_target_path) #.convert('RGB')
        lab_target = lab_target.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

        img_target, lab_target, scale = self.paired_transform(self.opt, img_target, lab_target)
        img_target = self.transform_no_augment(img_target)
        lab_target = self.transform_no_augment_depth(lab_target)

        return {'img_target': img_target, 'lab_target': lab_target,
                'img_target_paths': img_target_path, 'lab_target_paths': lab_target_path
                }

    def __len__(self):
        return self.img_target_size

    def _make_transform(self, opt, augment, depth=False):
        transforms_list = []

        if augment:
            if opt.isTrain:
                transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
        
        transforms_list.append(transforms.ToTensor())
        
        if depth:
            transforms_list.append(transforms.Normalize((0.5, ), (0.5, )))
        else:
            transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(transforms_list)

    def paired_transform(self, opt, image, depth):
        scale_rate = 1.0

        if opt.flip:
            n_flip = random.random()
            if n_flip > 0.5:
                image = F.hflip(image)
                depth = F.hflip(depth)

        if opt.rotation:
            n_rotation = random.random()
            if n_rotation > 0.5:
                degree = random.randrange(-500, 500)/100
                image = F.rotate(image, degree, Image.BICUBIC)
                depth = F.rotate(depth, degree, Image.BILINEAR)

        return image, depth, scale_rate


#####################################################################################################################


class CreateTaskDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(CreateTaskDataset, self).__init__()
        self.opt = opt

        self.img_source_paths, self.img_source_size = make_dataset(opt, opt.img_source_dir)  # img_source_dir = '^/trainA'
        self.img_target_paths, self.img_target_size = make_dataset(opt, opt.img_target_dir)  # img_source_dir = '^/trainB'

        self.lab_source_paths, self.lab_source_size = make_dataset(opt, opt.lab_source_dir)  # img_source_dir = '^/trainA_depth'
        # for visual results, not for training
        self.lab_target_paths, self.lab_target_size = make_dataset(opt, opt.lab_target_dir)  # img_source_dir = '^/trainB_depth'     
        
        if self.opt.isTrain:
            self.lab_source_paths, self.lab_source_size = make_dataset(opt, opt.lab_source_dir)  # img_source_dir = '^/trainA_depth'
            # for visual results, not for training
            self.lab_target_paths, self.lab_target_size = make_dataset(opt, opt.lab_target_dir)  # img_source_dir = '^/trainB_depth'

        self.transform_augment = self._make_transform(opt, True)
        self.transform_no_augment = self._make_transform(opt, False)  # for label

        self.transform_no_augment_depth = self._make_transform(opt, False, True)  # for label

    # get tensor data
    def __getitem__(self, item):
        index = random.randint(0, self.img_target_size - 1)
        img_source_path = self.img_source_paths[item % self.img_source_size]
        if self.opt.dataset_mode == 'paired':
            img_target_path = self.img_target_paths[item % self.img_target_size]
        elif self.opt.dataset_mode == 'unpaired':
            img_target_path = self.img_target_paths[index]
        else:
            raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

        img_source = Image.open(img_source_path).convert('RGB')
        img_target = Image.open(img_target_path).convert('RGB')
        img_source = img_source.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)
        img_target = img_target.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

        if self.opt.isTrain:
            lab_source_path = self.lab_source_paths[item % self.lab_source_size]
            if self.opt.dataset_mode == 'paired':
                lab_target_path = self.lab_target_paths[item % self.img_target_size]
            elif self.opt.dataset_mode == 'unpaired':
                lab_target_path = self.lab_target_paths[index]
            else:
                raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

            lab_source = Image.open(lab_source_path) #.convert('RGB')
            lab_target = Image.open(lab_target_path) #.convert('RGB')
            lab_source = lab_source.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)
            lab_target = lab_target.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

            img_source, lab_source, scale = self.paired_transform(self.opt, img_source, lab_source)
            img_source = self.transform_augment(img_source)
            lab_source = self.transform_no_augment_depth(lab_source)

            img_target, lab_target, scale = self.paired_transform(self.opt, img_target, lab_target)
            img_target = self.transform_no_augment(img_target)
            lab_target = self.transform_no_augment_depth(lab_target)

            return {'img_source': img_source, 'img_target': img_target,
                    'lab_source': lab_source, 'lab_target': lab_target,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    'lab_source_paths': lab_source_path, 'lab_target_paths': lab_target_path
                    }

        else:
            img_source = self.transform_augment(img_source)
            img_target = self.transform_no_augment(img_target)

            # return {'img_source': img_source, 'img_target': img_target,
            #         'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
            #         }


            ##############################################################################################
            lab_source_path = self.lab_source_paths[item % self.lab_source_size]
            if self.opt.dataset_mode == 'paired':
                lab_target_path = self.lab_target_paths[item % self.img_target_size]
            elif self.opt.dataset_mode == 'unpaired':
                lab_target_path = self.lab_target_paths[index]
            else:
                raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)
            lab_source = Image.open(lab_source_path) #.convert('RGB')
            lab_target = Image.open(lab_target_path) #.convert('RGB')
            lab_source = lab_source.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)
            lab_target = lab_target.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)
            lab_source = self.transform_no_augment_depth(lab_source)
            lab_target = self.transform_no_augment_depth(lab_target)

            return {'img_source': img_source, 'img_target': img_target,
                    'lab_source': lab_source, 'lab_target': lab_target,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    'lab_source_paths': lab_source_path, 'lab_target_paths': lab_target_path
                    }
            ##############################################################################################

    def __len__(self):
        return self.img_source_size

    def _make_transform(self, opt, augment, depth=False):
        transforms_list = []

        if augment:
            if opt.isTrain:
                transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
        
        transforms_list.append(transforms.ToTensor())
        
        if depth:
            transforms_list.append(transforms.Normalize((0.5, ), (0.5, )))
        else:
            transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(transforms_list)

    def paired_transform(self, opt, image, depth):
        scale_rate = 1.0

        if opt.flip:
            n_flip = random.random()
            if n_flip > 0.5:
                image = F.hflip(image)
                depth = F.hflip(depth)

        if opt.rotation:
            n_rotation = random.random()
            if n_rotation > 0.5:
                degree = random.randrange(-500, 500)/100
                image = F.rotate(image, degree, Image.BICUBIC)
                depth = F.rotate(depth, degree, Image.BILINEAR)

        return image, depth, scale_rate



class CreateMixedDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(CreateMixedDataset, self).__init__()
        self.opt = opt

        img_source_paths, self.img_source_size = make_dataset(opt, opt.img_source_dir)  # img_source_dir = '^/trainA'
        lab_source_paths, self.lab_source_size = make_dataset(opt, opt.lab_source_dir)  # img_source_dir = '^/trainA_depth'

        img_target_paths, self.img_target_size = make_dataset(opt, opt.img_target_dir)  # img_source_dir = '^/trainB'
        lab_target_paths, self.lab_target_size = make_dataset(opt, opt.lab_target_dir)  # img_source_dir = '^/trainB_depth'     
        
        for i in img_target_paths:
            img_source_paths.append(i)

        for t in lab_target_paths:
            lab_source_paths.append(t)

        self.img_source_paths = img_source_paths
        self.lab_source_paths = lab_source_paths

        self.transform_augment = self._make_transform(opt, True)
        self.transform_no_augment = self._make_transform(opt, False)  # for label

        self.transform_no_augment_depth = self._make_transform(opt, False, True)  # for label

    # get tensor data
    def __getitem__(self, item):
        img_source_path = self.img_source_paths[item % self.img_source_size]

        img_source = Image.open(img_source_path).convert('RGB')
        img_source = img_source.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

        lab_source_path = self.lab_source_paths[item % self.lab_source_size]

        lab_source = Image.open(lab_source_path) #.convert('RGB')
        lab_source = lab_source.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

        img_source, lab_source, scale = self.paired_transform(self.opt, img_source, lab_source)
        img_source = self.transform_augment(img_source)
        lab_source = self.transform_no_augment_depth(lab_source)

        return {'img_source': img_source, 'lab_source': lab_source, 
                'img_source_paths': img_source_path, 'lab_source_paths': lab_source_path
                }

    def __len__(self):
        return self.img_source_size + self.img_target_size

    def _make_transform(self, opt, augment, depth=False):
        transforms_list = []

        if augment:
            if opt.isTrain:
                transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
        
        transforms_list.append(transforms.ToTensor())
        
        if depth:
            transforms_list.append(transforms.Normalize((0.5, ), (0.5, )))
        else:
            transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(transforms_list)

    def paired_transform(self, opt, image, depth):
        scale_rate = 1.0

        if opt.flip:
            n_flip = random.random()
            if n_flip > 0.5:
                image = F.hflip(image)
                depth = F.hflip(depth)

        if opt.rotation:
            n_rotation = random.random()
            if n_rotation > 0.5:
                degree = random.randrange(-500, 500)/100
                image = F.rotate(image, degree, Image.BICUBIC)
                depth = F.rotate(depth, degree, Image.BILINEAR)

        return image, depth, scale_rate