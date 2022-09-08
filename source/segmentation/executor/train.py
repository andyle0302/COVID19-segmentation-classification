# https://github.com/qubvel/segmentation_models.pytorch
# https://github.com/IlliaOvcharenko/lung-segmentation
# https://www.kaggle.com/pezhmansamadi/lung-segmentation-torch

# import lib
import torch
import time
from operator import add
import sys

# import module
from dataloader.transform import augs, transfms


def train_loop(loader, set_model, metric_fn, device):

    epoch_loss = 0.0
    metrics_score = {
        'jaccard': 0.0,
        'acc': 0.0,
        'f1': 0.0,
        'recall': 0.0,
        'precision': 0.0}

    steps = len(loader)

    model = set_model['model']
    model.train()

    for i, (x, y, _, _) in enumerate(loader):
        x = x.to(device)
        y = y.float().unsqueeze(1).to(device)

        set_model['optimizer'].zero_grad()

        y_pred = model(x)  # gpu cuda

        loss = set_model['loss_fn'](y_pred, y)  # comboloss: BCE dice focal
        loss.backward()

        score = metric_fn(y_pred, y)
        # print('score = ',score)

        value = list(map(add, metrics_score.values(), score.values()))
        # print('value= ', value)
        metrics_score = dict(zip(metrics_score.keys(), value))
        # print('metrics_score = ', metrics_score)

        set_model['optimizer'].step()
        learning_rate = set_model['optimizer'].param_groups[0]['lr']

        epoch_loss += loss.item()

        sys.stdout.flush()
        sys.stdout.write(
            '\r Step: [%2d/%2d], loss: %.4f - acc: %.4f' % (
                i, steps, loss.item(), score['acc']))
    set_model['scheduler'].step()

    sys.stdout.write('\r')

    epoch_loss = epoch_loss/len(loader)

    for idx in range(len(value)):
        value[idx] /= len(loader)

    metrics_score = dict(zip(metrics_score.keys(), value))
    return {
        'loss': epoch_loss,
        'metrics_score': metrics_score,
        'learning_rate': learning_rate}


def evaluate(loader, set_model, metric_fn, device):
    epoch_loss = 0.0
    metrics_score = {
        'jaccard': 0.0,
        'acc': 0.0,
        'f1': 0.0,
        'recall': 0.0,
        'precision': 0.0}
    model = set_model['model']
    loss_fn = set_model['loss_fn']

    model.eval()
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)

            y = y.float().unsqueeze(1).to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            score = metric_fn(y_pred, y)
            value = list(map(add, metrics_score.values(), score.values()))
            metrics_score = dict(zip(metrics_score.keys(), value))

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)

    for idx in range(len(value)):
        value[idx] /= len(loader)

    metrics_score = dict(zip(metrics_score.keys(), value))

    return {'loss': epoch_loss, 'metrics_score': metrics_score}  # dict


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def fit(set_model, dataloader, metric_fn,  opt):
    """ fiting model to dataloaders,
    saving best weights and showing results """

    dataloader = dataloader(opt.data_path, opt.batch_size, augs, transfms)

    best_val_loss = float("inf")
    patience = 8
    train_list, val_list = [], []

    since = time.time()
    for epoch in range(opt.start_epoch, opt.end_epoch):
        ts = time.time()

        if opt.device is True:
            device = torch.device('cuda:0')

        train_dict = train_loop(
            dataloader['train'],
            set_model, metric_fn, device)

        val_dict = evaluate(
            dataloader['val'],
            set_model, metric_fn, device)

        te = time.time()

        epoch_mins, epoch_secs = epoch_time(ts, te)

        print(
            'Epoch [{}/{}], loss: {:.4f} - \
                jaccard: {:.4f} - acc: {:.4f}  - \
                f1: {:.4f} - recall: {:.4f} - \
                precision: {:.4f}'.format(
                    epoch + 1, opt.end_epoch, train_dict['loss'],
                    train_dict['metrics_score']['jaccard'],
                    train_dict['metrics_score']['acc'],
                    train_dict['metrics_score']['f1'],
                    train_dict['metrics_score']['recall'],
                    train_dict['metrics_score']['precision']))

        print(
            'val_loss: {:.4f} - val_jaccard: {:.4f} - \
            val_acc: {:.4f} - val_f1: {:.4f} - \
            val_recall: {:.4f} - val_precision: {:.4f}'.format(
                val_dict['loss'],
                val_dict['metrics_score']['jaccard'],
                val_dict['metrics_score']['acc'],
                val_dict['metrics_score']['f1'],
                val_dict['metrics_score']['recall'],
                val_dict['metrics_score']['precision']))

        print(f'Time: {epoch_mins}m {epoch_secs}s')

        train_list.append({
            'epoch': epoch,
            'train_loss': train_dict['loss'],
            'jaccard': train_dict['metrics_score']['jaccard'],
            'dice': train_dict['metrics_score']['f1'],
            'recall': train_dict['metrics_score']['recall'],
            'precision': train_dict['metrics_score']['precision'],
            'accuracy': train_dict['metrics_score']['acc']})

        val_list.append({
            'epoch': epoch,
            'val_loss': val_dict['loss'],
            'val_jaccard': val_dict['metrics_score']['jaccard'],
            'val_dice': val_dict['metrics_score']['f1'],
            'recall': val_dict['metrics_score']['recall'],
            'precision': val_dict['metrics_score']['precision'],
            'accuracy': val_dict['metrics_score']['acc']})

        period = time.time() - since
        print(
            'Training complete in {:.0f}m {:.0f}s'.format(
                period // 60, period % 60))

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': set_model['model'].state_dict(),
            'optimizer_state_dict': set_model['optimizer'].state_dict(),
            'train_list': train_list,
            'val_list': val_list,
            }, opt.checkpoint_path)

        if val_dict['loss'] < best_val_loss:
            count = 0
            data_str = f"===> Valid loss improved from {best_val_loss:2.4f} \
                to {val_dict['loss']:2.4f}. \
                Saving checkpoint: {opt.checkpoint_path}"
            print(data_str)
            best_val_loss = val_dict['loss']
            # save_checkpoint(model.state_dict(), checkpoint_path)

            # save model
            torch.save(set_model['model'].state_dict(), opt.model_path)

        else:
            count += 1
            if count >= patience:
                print('Early stopping!')
                return train_dict, val_dict

    return train_list, val_list
