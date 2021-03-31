import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse, time, sys, os, cv2
import numpy as np
from dataloader import AlzhDataset
import tensorboard_logger as tb_logger
from PIL import Image
from utils import AverageMeter, accuracy, adjust_learning_rate
from network.resnet import SupConResNet
from network.custom import Custom_CNN, Linear_cls
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default=[50,70,90],
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
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    train_dataset = AlzhDataset(root='/data/tm/alzh/data_PGGAN/train', transform=transform_train)
    valid_dataset = AlzhDataset(root='/data/tm/alzh/data_PGGAN/validation', transform=transform_train)
    # train_len = int(dataset.__len__() * 0.8)
    # valid_len = dataset.__len__() - train_len
    # train, val = torch.utils.data.random_split(dataset, [train_len, valid_len])
    train_loader = torch.utils.data.DataLoader(train_dataset, opt.batch_size, num_workers=0, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, opt.batch_size, num_workers=0, shuffle=False, drop_last=False)

    return train_loader, valid_loader

def set_model(opt):
    # model = Custom_CNN(in_channel=4)
    # model = torchvision.models.MobileNetV2(num_classes=2)
    # model = torchvision.models.resnet18(pretrained=True)
    # model.fc = nn.Linear(512, 1)
    model = SupConResNet(name='resnet18')
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt, model2):
    model.train()
    model2.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (image, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            image = image.cuda()
            labels = labels.float().cuda()
        bsz = labels.shape[0]
        logits = model.encoder(image)
        logits = model2(logits)
        loss = criterion(logits.squeeze(1), labels)
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

def validation(val_loader, model, model2):
    model.eval()
    model2.eval()
    top1 = AverageMeter()
    with torch.no_grad():
        for i, (image, label) in enumerate(val_loader):
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()
            bsz = label.shape[0]
            output = model.encoder(image)
            output = model2(output).squeeze(1)
            acc1 = 100 * torch.eq(torch.round(output), label).sum().item() / bsz
            # acc1 = accuracy(output, label)
            top1.update(acc1, bsz)
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def test(type, model, model2, transform=None, root='/data/tm/alzh/data_PGGAN/test'):
    model.eval()
    model2.eval()
    correct = 0
    total = 0
    correct_image = 0
    total_image = 0
    with torch.no_grad():
        path = os.path.join(root, type)
        for dirname in os.listdir(path):
            new_path = os.path.join(path, dirname)
            for i in range(len(os.listdir(new_path))):
                img = Image.open(os.path.join(new_path, os.listdir(new_path)[i])).convert('RGB')
                # canny = cv2.Canny(np.asarray(img), 50, 150, L2gradient=True) / 255.
                if transform is not None:
                    img = transform(img)
                # img = torch.cat([img, torch.from_numpy(canny).unsqueeze(0)], dim=0).float()
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
            output = model.encoder(img_concat)
            output = model2(output).squeeze(1)
            acc1 = 100 * torch.eq(torch.round(output), label).sum().item() / bsz
            # acc1 = accuracy(output, labels)
            if acc1 > 50:
                correct += 1
            total += 1
            correct_image += torch.eq(torch.round(output), label).sum().item()
            total_image += bsz
    return total, correct, total_image, correct_image

def main():
    opt = parse_option()
    train_loader, valid_loader = set_loader(opt)
    model, criterion = set_model(opt)
    checkpoint = torch.load('./save_models/SimCLR-CBAM/800.pth')
    model.load_state_dict(checkpoint)
    model2 = Linear_cls(512, 1)
    model2 = model2.cuda()
    optimizer = torch.optim.SGD(list(model.parameters()) +list(model2.parameters()),
                                lr=opt.learning_rate,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    valid_best = 0
    best_epoch = 0
    for epoch in range(1, opt.epochs + 1):
        lr = adjust_learning_rate(opt, optimizer, epoch)
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt, model2)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        top1 = validation(valid_loader, model, model2)
        # if top1 > valid_best:
        #     valid_best = top1
        #     best_epoch = epoch
        #     torch.save(model.state_dict(), './best_model.pth')
        AD_total, AD_correct, AD_total_image, AD_correct_image = test('AD', model, model2, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ]))
        CN_total, CN_correct, CN_total_image, CN_correct_image = test('CN', model, model2, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))
        test_acc = 100 * (AD_correct_image + CN_correct_image) / (AD_total_image + CN_total_image)
        print('AD: {0}/{1}\t'
              'CN: {2}/{3}\t'
              'test acc {4}\t'.format(AD_correct, AD_total, CN_correct, CN_total, test_acc))
        logger.log_value('loss', loss, epoch)
        logger.log_value('valid acc', top1, epoch)
        logger.log_value('test acc', test_acc, epoch)
        logger.log_value('AD test acc', AD_correct/AD_total, epoch)
        logger.log_value('CN test acc', CN_correct/CN_total, epoch)


if __name__ == '__main__':
    main()
