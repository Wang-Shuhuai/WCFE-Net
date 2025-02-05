import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed

#sys.path.append("../")
from utils import *
from torchvision import datasets, transforms
from torch.autograd import Variable
from birealnet import birealnet18, birealnet34, BinarizeConv2d
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(os.path.join(args.results_dir, 'runs/AbWb_Jump_C_0.0005_100grad_ep100'))
writer = SummaryWriter('/root/tf-logs/recu_SA_bb_Jump_emagradL360S1800_ep120_wdf0b5e-5_sgd_noalpha_Nbias_b256_0.25_seed1')

def statistic(model, total_iter):
    index = 0
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            x = m.weight.data[1, 1, 1, 1]
            y = m.weight.grad.data[1, 1, 1, 1]
            writer.add_scalar('weight/{}'.format(index), x, total_iter)
            writer.add_scalar('weight_grad/{}'.format(index), y, total_iter)
            index += 1


class BinOp():
    def __init__(self, model):
        # count the number of Conv2d

        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, BinarizeConv2d):
                count_Conv2d = count_Conv2d + 1

        self.num_of_params = count_Conv2d
        self.saved_params = []
        self.target_modules = []
        self.clip_th = 0
        self.grad_normal_ema = []
        self.current_ep = 0
        self.flip_count = 0

        for m in model.modules():
            if isinstance(m, BinarizeConv2d):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp)
                self.target_modules.append(m.weight)
                self.grad_normal_ema.append(0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def calculate_flip(self, ep, lr):
        if self.current_ep != ep:
            self.current_ep = ep
            writer.add_scalar('flip_count', self.flip_count / 5000, ep - 1)
            writer.add_scalar('PerEpoch total_th', self.clip_th / 5000, ep - 1)
            self.flip_count = 0
            self.clip_th = 0

        for index in range(self.num_of_params):
            beta = 0.99
            sth = torch.abs(self.target_modules[index].grad.data).sum() / self.target_modules[index].grad.data.nelement()

            self.grad_normal_ema[index] = beta * self.grad_normal_ema[index] + (1 - beta) * sth

            gne_correct = self.grad_normal_ema[index] / (1 - (beta) ** (total_iter + 1))

            # writer.add_scalar('ema_gradnormal/{}'.format(index), gne_correct, total_iter)
            #
            self.target_modules[index].data[(self.saved_params[index].sign() != self.target_modules[index].data.sign())] = \
                self.target_modules[index].data[(self.saved_params[index].sign() != self.target_modules[index].data.sign())].sign() * lr * gne_correct * 1800
            self.target_modules[index].data = self.target_modules[index].data.clamp(-360*gne_correct, 360*gne_correct)

            self.flip_count += (self.saved_params[index].sign() != self.target_modules[index].data.sign()).sum()
            self.clip_th += sth

parser = argparse.ArgumentParser("birealnet")
parser.add_argument('--seed', default=1, type=int, help='random seed, set to 0 to disable')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=256, help='num of training epochs')
parser.add_argument('--epochs1', type=int, default=120, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay')
parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()

CLASSES = 1000

if not os.path.exists('log'):
    os.mkdir('log')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('log/log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()
    #
    # cudnn.benchmark = True
    # cudnn.enabled=True
    logging.info("args = %s", args)

    if args.seed > 0:
        set_seed(args.seed)
    else:
        cudnn.benchmark = True
        cudnn.enabled=True

    # load model
    model = birealnet34()
    logging.info(model)
    model = nn.DataParallel(model).cuda()

    bin_op = BinOp(model)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or pname=='classifier.0.weight' or pname == 'classifier.0.bias':
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.SGD([{'params': other_parameters,'initial_lr':args.learning_rate,'weight_decay':0},
                                 {'params': weight_parameters, 'weight_decay' : args.weight_decay}],
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optimizer = torch.optim.Adam(
    #         [{'params' : other_parameters},
    #         {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
    #         lr=args.learning_rate,)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs1, eta_min=0, last_epoch=-1)
    start_epoch = 0
    best_top1_acc= 0
    #
    # checkpoint_tar = os.path.join(args.save, 'checkpoint.pth.tar')
    # if os.path.exists(checkpoint_tar):
    #     logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
    #     checkpoint = torch.load(checkpoint_tar)
    #     start_epoch = checkpoint['epoch']
    #     best_top1_acc = checkpoint['best_top1_acc']
    #     model.load_state_dict(checkpoint['state_dict'], strict=False)
    #     logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))

    # adjust the learning rate according to the checkpoint
    for epoch in range(start_epoch):
        scheduler.step()

    # load training data
    traindir = os.path.join( '/root/autodl-tmp/imagenet/train')
    valdir = os.path.join('/root/autodl-tmp/imagenet/val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data augmentation
    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # load validation data
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # train the model
    epoch = start_epoch
    while epoch < args.epochs1:
        train_obj, train_top1_acc,  train_top5_acc = train(bin_op, epoch,  train_loader, model, criterion_smooth, optimizer, scheduler)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.save)

        epoch += 1

    training_time = (time.time() - start_t) / 36000
    print('total training time = {} hours'.format(training_time))

total_iter = 0
def train(bin_op, epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    global total_iter
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()


    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        bin_op.save_params()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bin_op.calculate_flip(epoch, optimizer.param_groups[0]['lr'])
        # statistic(model, total_iter)
        total_iter += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i%500==0:
            progress.display(i)

    scheduler.step()
    writer.add_scalar('train/train_prec1', top1.avg, epoch)
    writer.add_scalar('train/model_lr', optimizer.param_groups[0]['lr'], epoch)
    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
        writer.add_histogram("{}/{} grad".format(layer, attr), param.grad, epoch)
    return losses.avg, top1.avg, top5.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if(i%50==0):
                progress.display(i)

        print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    writer.add_scalar('train/val_prec1', top1.avg, epoch)
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
