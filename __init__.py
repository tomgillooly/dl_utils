import argparse
import datetime
import numpy as np
import os
import subprocess
import torch

from collections import defaultdict
from zlib import crc32


class BaseParser(argparse.ArgumentParser):
    def __init__(self):
        super(BaseParser, self).__init__()

        self.add_argument('--name', type=str, default='sandbox')
        self.add_argument('--dataroot', type=str, default='/home/tom/cluster/Matterport3DSimulator/data/dumb_trajectories')
        self.add_argument('--phase', type=str, default='train')
        self.add_argument('--batch_size', type=int, default=1)
        self.add_argument('--num_workers', type=int, default=4)
        self.add_argument('--batch_repeats', type=int, default=4)
        self.add_argument('--learning_rate', type=float, default=0.01)
        self.add_argument('--beta1', type=float, default=0.9)
        self.add_argument('--beta2', type=float, default=0.999)

        self.add_argument('--n_epochs', type=int, default=5)
        self.add_argument('--start_epoch', type=int, default=1)
        self.add_argument('--load_epoch', type=str, default='')

        self.add_argument('--print_every', type=int, default=5000)
        self.add_argument('--plot_every', type=int, default=100)
        self.add_argument('--save_every', type=int, default=20000)
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
                    file.write('{}:{}\n'.format(name, val))

            with open(os.path.join(args.save_dir, args.name, git_patch_name), 'w') as file:
                file.write(subprocess.check_output('git diff HEAD'.split(), universal_newlines=True))

        return args


class DummyVisdom(object):
    def __init__(self, *args, **kwargs):
        pass

    def add_plot_data(self, name, data, epoch, iter):
        pass

    def plot_line(self):
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

        self.running_plot_data = defaultdict(list)
        self.plot_data = defaultdict(list)

        self.base_win_id = crc32(bytes(args.name, encoding='latin1'))

    def add_plot_data(self, name, data, epoch, iter):
        self.running_plot_data[name].append((epoch + float(iter) / self.total_epoch_iters, data))

    def process_plot_data(self):
        for series_name, points in self.running_plot_data.items():
            if len(points) == 0:
                continue
            x_data, y_data = zip(*points)
            avg_data = np.mean(torch.Tensor(y_data).clone().detach().cpu().numpy())
            self.plot_data[series_name].append((x_data[-1], avg_data))
            self.running_plot_data[series_name] = []

    def plot_line(self):
        self.process_plot_data()

        for i, (series_name, points) in enumerate(self.plot_data.items()):
            x_data, y_data = zip(*points)
            self.vis.line(X=torch.Tensor(x_data).clone().detach().cpu(),
                          Y=torch.Tensor(y_data).clone().detach().cpu(),
                          win=self.base_win_id+1+i,
                          opts={'title': '{} {}'.format(self.args.name, series_name.lower())})

    def print_plot_data(self, epoch, iter):
        self.process_plot_data()

        message = 'Epoch {}, iter {}: '.format(epoch, iter) + \
                  ', '.join(['{} - {}'.format(series_name, values[-1][1])
                             for series_name, values in self.plot_data.items()])

        with open(self.args.log_file_name, 'a') as file:
            file.write(message + '\n')

        print(message)


class BaseModel(object):
    def forward(self, input):
        raise NotImplementedError('Model forward pass not implemented')

    def zero_optimisers(self):
        raise NotImplementedError('Model zero optimisers not implemented')

    def step_optimisers(self):
        raise NotImplementedError('Model step optimisers not implemented')

    def get_metrics(self):
        raise NotImplementedError('Model get metrics not implemented')

    def to(self, device):
        raise NotImplementedError('Model to-device not implemented')

    def train(self):
        raise NotImplementedError('Model train method not implemented')

    def eval(self):
        raise NotImplementedError('Model eval method not implemented')


def train(args, model, train_loader, validation_loader):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(model)

    model = model.to(device)

    plotter = Plotter(args, len(train_loader.dataset))

    if args.load_epoch != '':
        # Convert load_iter to correctly formatted string if necessary
        try:
            args.load_epoch = '{:06}'.format(int(args.load_epoch))
        except ValueError:
            pass

        state_dict = torch.load(os.path.join('checkpoints', args.name,
                                             '{}_{}.pth'.format(model.save_name, args.load_epoch)))

        if 'stop_epoch' in state_dict.keys():
            args.load_epoch = state_dict.pop('stop_epoch')+1
            print('Loading step {}'.format(args.load_epoch))

        model.load_state_dict(state_dict)
    else:
        args.load_epoch = 0

    train_step = 0

    train_load_iter = iter(train_loader)
    validation_load_iter = iter(validation_loader)

    for epoch in range(args.load_epoch, args.n_epochs):
        for epoch_step, data in enumerate(train_load_iter):
            model.train()
            model.zero_optimisers()

            loss = model.forward(data)

            loss.backward()

            model.step_optimisers()

            plotter.add_plot_data(('loss', loss.item(), epoch, epoch_step))
            for key, value in model.get_metrics().items():
                plotter.add_plot_data(key, value, epoch, epoch_step)

            model.eval()
            try:
                data = next(validation_load_iter)
            except StopIteration:
                validation_load_iter = iter(validation_loader)
                data = next(validation_load_iter)

            loss = model.forward(data)
            plotter.add_plot_data(('validation_loss', loss.item(), epoch, epoch_step))
            for key, value in model.get_metrics().items():
                plotter.add_plot_data('validation_' + key, value, epoch, epoch_step)

            if (train_step+1) % args.print_every == 0:
                plotter.print_plot_data(epoch, epoch_step)
                plotter.plot_line()

            if (train_step+1) % args.save_every == 0:
                state_dict = model.state_dict()
                state_dict['stop_epoch'] = epoch
                torch.save(state_dict,
                           os.path.join(args.save_dir, args.name, '{}_{:06}.pth'.format(model.save_name, train_step+1)))

            train_step += 1
