import argparse
import datetime
import numpy as np
import os
import subprocess
import torch
import torch.nn as nn

from collections import defaultdict, OrderedDict
from hashlib import md5
from zlib import crc32

from . import metrics


def list_collate(batch):
    output = {}
    for key in batch[0].keys():
        output[key] = [data[key] for data in batch]

    return output


class BaseParser(argparse.ArgumentParser):
    def __init__(self):
        super(BaseParser, self).__init__()

        self.add_argument('--name', type=str, default='sandbox')
        self.add_argument('--dataroot', type=str, default='/home/tom/cluster/Matterport3DSimulator/data/dumb_trajectories')
        self.add_argument('--gt_dataroot', type=str, default='/home/tom/cluster/Matterport3DSimulator/data/dumb_trajectories')
        self.add_argument('--phase', type=str, default='train')
        self.add_argument('--batch_size', type=int, default=1)
        self.add_argument('--num_workers', type=int, default=4)
        self.add_argument('--batch_repeats', type=int, default=4)
        self.add_argument('--learning_rate', type=float, default=0.01)
        self.add_argument('--beta1', type=float, default=0.9)
        self.add_argument('--beta2', type=float, default=0.999)
        self.add_argument('--weight_decay', type=float, default=1e-5)

        training_duration = self.add_mutually_exclusive_group()
        training_duration.add_argument('--n_epochs', type=int)
        training_duration.add_argument('--n_iters', type=int)
        self.add_argument('--start_epoch', type=int, default=1)
        self.add_argument('--load_epoch', type=str, default='')

        self.add_argument('--print_every', type=int, default=5000)
        self.add_argument('--plot_every', type=int, default=100)
        self.add_argument('--save_every', type=int, default=20000)
        self.add_argument('--train_eval_every', type=int, default=5)
        self.add_argument('--save_dir', type=str, default='checkpoints')

        self.add_argument('--visdom_host', type=str, default='localhost')
        self.add_argument('--visdom_port', type=int, default=8097)

    def parse_args(self, *args):
        args = super(BaseParser, self).parse_args(*args)

        now_string = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')

        if args.phase == 'train':
            try:
                args.slurm_job_id = os.environ['SLURM_JOB_ID']
                git_patch_name = args.slurm_job_id + '.patch'
            except KeyError:
                args.slurm_job_id = 'None'
                git_patch_name = 'job_' + now_string + '.patch'

            args.git_commit = subprocess.check_output('git rev-parse HEAD'.split(), universal_newlines=True)

            os.makedirs(os.path.join(args.save_dir, args.name), exist_ok=True)

            args.log_file_name = os.path.join(args.save_dir, args.name, 'log_{}.log'.format(now_string))

            with open(os.path.join(args.save_dir, args.name, 'opts_{}.txt'.format(now_string)), 'w') as file:
                for name, val in sorted(vars(args).items()):
                    print('{}:{}'.format(name, val))
                    file.write('{}:{}\n'.format(name, val))

            with open(os.path.join(args.save_dir, args.name, git_patch_name), 'w') as file:
                file.write(subprocess.check_output('git diff HEAD'.split(), universal_newlines=True))

        return args


class DummyVisdom(object):
    def __init__(self, *args, **kwargs):
        pass

    def line(self, *args, **kwargs):
        pass


class Plotter(object):
    def __init__(self, args, dataset_len):
        if args.visdom_host == 'None':
            self.vis = DummyVisdom()
        else:
            import visdom
            self.vis = visdom.Visdom(server=args.visdom_host, port=args.visdom_port)

        self.args = args
        self.total_epoch_iters = dataset_len / args.batch_size

        self.running_plot_data = OrderedDict()
        self.plot_data = defaultdict(list)

        self.base_win_id = crc32(bytes(args.name, encoding='latin1'))
        self.window_ids = {}
        self.legend = defaultdict(list)

    def get_window_id(self, series_name):
        key = series_name.replace('_train', '').replace('_validation', '')

        if key not in self.window_ids.keys():
            self.window_ids[key] = str(self.base_win_id + len(self.window_ids.items()))

        return self.window_ids[key]

    def add_plot_data(self, name, data, epoch, iter):
        if name not in self.running_plot_data.keys():
            self.running_plot_data[name] = []
        self.running_plot_data[name].append((epoch + float(iter) / self.total_epoch_iters, data))

    def process_plot_data(self):
        for series_name, points in self.running_plot_data.items():
            if len(points) == 0:
                continue
            x_data, y_data = zip(*points)
            avg_data = np.mean(torch.Tensor(y_data).clone().detach().cpu().numpy())
 
            win_id = self.get_window_id(series_name)
            if series_name not in self.legend[win_id]:
                self.legend[win_id].append(series_name)

            self.plot_data[series_name] = ([x_data[-1]], [avg_data])

            self.running_plot_data[series_name] = []

    def plot_line(self):
        self.process_plot_data()

        for key in self.window_ids.keys():
            legend = []
            x_data = 0
            y_data = []
            for series_name, points in filter(lambda item: key in item[0], self.plot_data.items()):
                x, y = points
                x_data = max(x[0], x_data)
                y_data.append(y)
                legend.append(series_name)
           
            win_id = self.get_window_id(series_name)
            self.vis.line(X=torch.Tensor([x_data]).clone().detach().cpu(),
                          Y=torch.Tensor(y_data).clone().detach().cpu(),
                          win=win_id,
                          update='append',
                          opts={'title': '{} {}'.format(self.args.name, key),
                                'legend': legend})
        self.plot_data = defaultdict(list)


    def print_plot_data(self, epoch, iter):
        self.process_plot_data()

        message = 'Epoch {}, iter {}: '.format(epoch, iter) + \
                  ', '.join(['{} - {}'.format(series_name, values[1][-1])
                             for series_name, values in self.plot_data.items()])

        with open(self.args.log_file_name, 'a') as file:
            file.write(message + '\n')

        print(message)


class BaseModel(nn.Module):
    def forward(self, input):
        raise NotImplementedError('Model forward pass not implemented')

    def zero_optimisers(self):
        [optimiser.zero_grad() for optimiser in self.optimisers]

    def step_optimisers(self):
        [optimiser.step() for optimiser in self.optimisers]

    def get_metrics(self, input, output):
        raise NotImplementedError('Model get metrics not implemented')

    def to(self, device):
        super(BaseModel, self).to(device)
        self.device = device

        return self

    def load_state_dict(self, state_dict):
        if 'stop_epoch' in state_dict.keys():
            state_dict.pop('stop_epoch')
        return nn.Module.load_state_dict(self, state_dict)

    def step_schedulers(self):
        pass


def get_hash(filename):
    m = md5()
    m.update(filename.encode())

    return m.digest()

def full_dataset_eval(model, loader, dataset_name, plotter, epoch, train_step):
    message = ''
    model.eval()

    load_iter = iter(loader)
    total_loss = 0
    metrics = defaultdict(float)
    for i, data in enumerate(load_iter):
        print('Processing {} of {}\r\r'.format(i*loader.batch_size, len(loader.dataset)), end='\r')
        with torch.no_grad():
            batch_output = model.forward(data)
            batch_loss = model.criterion(data, batch_output).item()
            total_loss += batch_loss
            # print('{}_batch_loss {}'.format(dataset_name, batch_loss))
        for key, value in model.get_metrics(data, batch_output).items():
            # print('{}_batch {} - {}'.format(dataset_name, key, value.item()))
            metrics[key] += value.item()
    total_loss /=  (len(loader.dataset) // loader.batch_size)
    message += ', {}_loss - {}'.format(dataset_name, total_loss)
    plotter.add_plot_data('loss_{}'.format(dataset_name), total_loss, epoch, train_step)
    for key, value in metrics.items():
        value /= (len(loader.dataset) // loader.batch_size)
        message += ', {}_{} - {}'.format(dataset_name, key, value)
        plotter.add_plot_data('{}_{}'.format(key, dataset_name), value, epoch, train_step)

    return message


def train(args, model, train_loader, validation_loader):
    #train_set = set([data['data_hash'] for data in train_loader.dataset])
    #validation_set = set([data['data_hash'] for data in validation_loader.dataset])

    #assert len(train_set.intersection(validation_set)) == 0

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(model)

    print('Loaded {} training data points'.format(len(train_loader.dataset)))
    print('Loaded {} validation data points'.format(len(validation_loader.dataset)))

    model = model.to(device)

    plotter = Plotter(args, len(train_loader.dataset))

    train_step = 0
    
    if args.load_epoch != '':
        # Convert load_iter to correctly formatted string if necessary
        try:
            args.load_epoch = '{:06}'.format(int(args.load_epoch))
        except ValueError:
            pass

        state_dict = torch.load(os.path.join('checkpoints', args.name,
                                             '{}_{}.pth'.format(model.save_name, args.load_epoch)))

        if 'stop_epoch' in state_dict.keys():
            stop_point = state_dict.pop('stop_epoch')
            try:
                args.load_epoch, train_step = stop_point
            except TypeError:
                if args.load_epoch == 'latest':
                    args.load_epoch = stop_point+1
                    # Best estimate of stopping point
                    train_step = (len(train_loader.dataset) // args.batch_size)*stop_point + 1
                else:
                    train_step = int(args.load_epoch)+1 
                    args.load_epoch = stop_point + 1
            print('Loading epoch {}'.format(args.load_epoch))

        model.load_state_dict(state_dict)
    else:
        args.load_epoch = 0

    epoch_start_time = datetime.datetime.now()
    model.zero_optimisers()

    # Get final epoch based on number of desired training iterations
    iters_per_epoch = len(train_loader.dataset) / args.batch_size
    iters_per_epoch = int(iters_per_epoch) if train_loader.drop_last else int(np.ceil(iters_per_epoch))
    train_end_epoch = int(np.ceil(args.n_iters / iters_per_epoch)) if args.n_iters else args.n_epochs

    for epoch in range(args.load_epoch, train_end_epoch):
        epoch_length = datetime.datetime.now() - epoch_start_time
        message = 'Start of epoch {}, duration {}:{}:{}'.format(epoch, epoch_length.seconds//3600,
                                                                (epoch_length.seconds//60)%60, epoch_length.seconds%60)
        message += full_dataset_eval(model, validation_loader, 'validation', plotter, epoch, 0)
        print(message)

        epoch_start_time = datetime.datetime.now()
        train_load_iter = iter(train_loader)
        for epoch_step, data in enumerate(train_load_iter):
            model.train()

            output = model.forward(data)
            loss = model.criterion(data, output)

            plotter.add_plot_data('loss', loss.item(), epoch, epoch_step)

            loss /= args.batch_repeats

            loss.backward()

            if (train_step+1) % args.batch_repeats == 0:
                model.step_optimisers()
                model.zero_optimisers()

            for key, value in model.get_metrics(data, output).items():
                plotter.add_plot_data(key, value, epoch, epoch_step)

            if (train_step+1) % args.print_every == 0:
                plotter.print_plot_data(epoch, train_step)
                plotter.plot_line()

            if (train_step+1) % args.save_every == 0:
                state_dict = model.state_dict()
                state_dict['stop_epoch'] = (epoch, train_step+1)
                torch.save(state_dict,
                           os.path.join(args.save_dir, args.name, '{}_{:06}.pth'.format(model.save_name, train_step+1)))

            model.step_schedulers()
            train_step += 1

            if args.n_iters is not None and train_step == args.n_iters:
                break

        state_dict = model.state_dict()
        state_dict['stop_epoch'] = (epoch, train_step+1)
        torch.save(state_dict,
                   os.path.join(args.save_dir, args.name, '{}_latest.pth'.format(model.save_name)))
        print('Saving latest model')
        model.print_optim_params()

        if (epoch + 1) % args.train_eval_every == 0:
            message = 'End of epoch {}'.format(epoch)
            message += full_dataset_eval(model, train_loader, 'train', plotter, epoch+1, 0)

            print(message)
