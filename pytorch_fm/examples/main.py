# -*- coding:utf-8 -*-
# -*- @author：hanyan5
# -*- @date：2020/10/22 19:12
# -*- python3.6
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from torchfm.model.lr import LogisticRegressionModel
import argparse


# 获取数据集
def get_dataset(name, path):
    if name == 'movielens1m':
        return MovieLens1MDataset(path)
    elif name == 'movielens20m':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


# 获取模型
def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    else:
        raise ValueError('unknown model name: ' + name)


# 训练
def train(model, optimizer, data_loader, criterion, device, log_interval=1000):
    model.train()
    total_loss = 0
    for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        fields, target = fields.to(device), target.to(device)
        y = model(fields.to(torch.long))
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            print('    - loss:', total_loss / log_interval)
            total_loss = 0


# 测试
def test(model, dataloader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields.to(torch.long))
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    return roc_auc_score(targets, predicts)


# 定义run函数
def run(dataset_name, dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length

    # 训练集，验证集，测试集
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, (
    train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    # 获取模型及field_nums
    model = get_model(model_name, dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch_i in range(epoch):
        # 训练
        train(model, optimizer, train_data_loader, criterion, device)
        # 验证
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation auc:', auc)
    # 测试
    auc = test(model, test_data_loader, device)
    print('test auc:', auc)
    # 保存模型
    torch.save(model, f'{save_dir}/{model_name}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1m')
    parser.add_argument('--dataset_path', help='criteo/tarin.txt, avazu/train, or ml-1m/ratings.dat', default='ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='lr')
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default='model_result')
    args = parser.parse_args()
    run(args.dataset_name, args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size,
        args.weight_decay, args.device, args.save_dir)
