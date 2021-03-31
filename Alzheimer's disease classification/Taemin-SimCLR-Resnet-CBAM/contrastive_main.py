import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse, time, sys, os, cv2
import numpy as np
from contrastive_dataloader import AlzhDataset
import tensorboard_logger as tb_logger
from tensorboardX import SummaryWriter
from PIL import Image
from utils import AverageMeter, accuracy, adjust_learning_rate
from network.resnet import SupConResNet
# from network.CBAM_resnet import SupConResNet
from network.custom import Custom_CNN, Linear_cls
from network.U_Net import UNet
from SupCon.losses import SupConLoss
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default=[700,800,900],
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    opt = parser.parse_args()

    opt.tb_path = './logs'
    opt.tb_folder = os.path.join(opt.tb_path, time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt
def set_loader(opt):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = [
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        # ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ]
    # transform_train = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.ToTensor(),
    # ])
    train_dataset = AlzhDataset(root='/data/tm/alzh/data_PGGAN/train', transform=transforms.Compose(train_transform))
    valid_dataset = AlzhDataset(root='/data/tm/alzh/data_PGGAN/validation', transform=transforms.Compose(train_transform))
    train_loader = torch.utils.data.DataLoader(train_dataset, opt.batch_size, num_workers=0, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, opt.batch_size, num_workers=0, shuffle=False, drop_last=False)

    return train_loader, valid_loader

def set_model(opt):
    # model = Custom_CNN(in_channel=4)
    # model = torchvision.models.MobileNetV2(num_classes=2)
    # model = torchvision.models.resnet18(pretrained=True)
    # model.fc = nn.Linear(512, 1)
    model = SupConResNet(name='resnet18')
    print(model)
    # criterion = nn.CrossEntropyLoss()
    # model = UNet(3, 2)
    # criterion = nn.L1Loss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    criterion = SupConLoss()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (image1, image2, label) in enumerate(train_loader):
        data_time.update(time.time() - end)
        image = torch.cat([image1, image2], dim=0)
        if torch.cuda.is_available():
            image = image.cuda()
        bsz = image1.shape[0]

        f = model(image)
        f1, f2 = torch.split(f, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg

def validation(val_loader, model, criterion):
    model.eval()
    top1 = AverageMeter()
    with torch.no_grad():
        for idx, (image1, image2, label) in enumerate(val_loader):
            image = torch.cat([image1, image2], dim=0)
            if torch.cuda.is_available():
                image = image.cuda()
            bsz = image1.shape[0]

            f = model(image)
            f1, f2 = torch.split(f, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features)
            top1.update(loss.item(), bsz)
        print(' * Loss {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def test(type, model, transform=None, root='/data/tm/alzh/data_PGGAN/test'):
    model.eval()
    correct = 0
    total = 0
    acc = 0
    with torch.no_grad():
        path = os.path.join(root, type)
        for dirname in os.listdir(path):
            new_path = os.path.join(path, dirname)
            for i in range(len(os.listdir(new_path))):
                img = Image.open(os.path.join(new_path, os.listdir(new_path)[i])).convert('RGB')
                canny = cv2.Canny(np.asarray(img), 50, 150, L2gradient=True) / 255.
                if transform is not None:
                    img = transform(img)
                img = torch.cat([img, torch.from_numpy(canny).unsqueeze(0)], dim=0).float()
                if i == 0:
                    img_concat = img.unsqueeze(0)
                else:
                    img_concat = torch.cat([img_concat, img.unsqueeze(0)], dim=0)
            if type is 'AD':
                label = torch.zeros((i + 1))
            else:
                label = torch.ones((i + 1))

            if torch.cuda.is_available():
                img_concat = img_concat.cuda()
                label = label.cuda()
            bsz = label.shape[0]
            output = model(img_concat)
            acc1 = 100 * torch.eq(torch.round(output), label).sum().item() / bsz
            # acc1 = accuracy(output, labels)
            if acc1 > 50:
                correct += 1
            total += 1
            acc += (acc1 / 10)
    return total, correct, acc

def main():
    opt = parse_option()
    train_loader, valid_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt.learning_rate,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
    writer = SummaryWriter(opt.tb_folder)
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    valid_best = 0
    best_epoch = 0
    for epoch in range(1, opt.epochs + 1):
        lr = adjust_learning_rate(opt, optimizer, epoch)
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        top1 = validation(valid_loader, model, criterion)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), './save_models/SimCLR-CBAM/'+str(epoch)+'.pth')
        # if top1 > valid_best:
        #     valid_best = top1
        #     best_epoch = epoch
        #     torch.save(model.state_dict(), './best_model.pth')
        # AD_total, AD_correct, AD_acc = test('AD', model, transform=transforms.Compose([
        # transforms.Resize(256),
        # transforms.ToTensor(),
        # ]))
        # CN_total, CN_correct, CN_acc = test('CN', model, transform=transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.ToTensor(),
        # ]))
        # test_acc = (AD_acc + CN_acc) / (10 * (AD_total + CN_total))
        # print('AD: {0}/{1}\t'
        #       'CN: {2}/{3}\t'
        #       'test acc {4}\t'.format(AD_correct, AD_total, CN_correct, CN_total, test_acc))
        writer.add_scalars('Train/loss', {'train': loss}, epoch)
        writer.add_scalars('Val/loss', {'valid': top1}, epoch)
        # logger.log_value('test acc', test_acc, epoch)
        # logger.log_value('AD test acc', AD_correct/AD_total, epoch)
        # logger.log_value('CN test acc', CN_correct/CN_total, epoch)


if __name__ == '__main__':
    main()
