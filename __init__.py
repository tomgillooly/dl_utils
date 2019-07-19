import argparse
import datetime
import numpy as np
import os
import subprocess
import torch

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
        [optimiser.zero_grad() for optimiser in self.optimisers]

    def step_optimisers(self):
        [optimiser.step() for optimiser in self.optimisers]

    def get_metrics(self):
        raise NotImplementedError('Model get metrics not implemented')

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device

        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        if 'stop_epoch' in state_dict.keys():
            state_dict.pop('stop_epoch')
        return self.model.load_state_dict(state_dict)

    def __repr__(self):
        return repr(self.model)


def get_hash(filename):
    m = md5()
    m.update(filename.encode())

    return m.digest()


def train(args, model, train_loader, validation_loader):
    train_set = set([data['data_hash'] for data in train_loader.dataset])
    validation_set = set([data['data_hash'] for data in validation_loader.dataset])

    assert len(train_set.intersection(validation_set)) == 0

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(model)

    print('Loaded {} training data points'.format(len(train_loader.dataset)))

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

    model.zero_optimisers()
    for epoch in range(args.load_epoch, args.n_epochs):
        epoch_start_time = datetime.datetime.now()
        train_load_iter = iter(train_loader)
        for epoch_step, data in enumerate(train_load_iter):
            model.train()

            loss = model.forward(data) / args.batch_repeats

            loss.backward()

            if (train_step+1) % args.batch_repeats == 0:
                model.step_optimisers()
                model.zero_optimisers()

            plotter.add_plot_data('loss', loss.item(), epoch, train_step)
            for key, value in model.get_metrics().items():
                plotter.add_plot_data(key, value, epoch, train_step)

            if (train_step+1) % args.print_every == 0:
                plotter.print_plot_data(epoch, train_step)
                plotter.plot_line()

            if (train_step+1) % args.save_every == 0:
                state_dict = model.state_dict()
                state_dict['stop_epoch'] = epoch
                torch.save(state_dict,
                           os.path.join(args.save_dir, args.name, '{}_{:06}.pth'.format(model.save_name, train_step+1)))

            train_step += 1

        epoch_length = datetime.datetime.now() - epoch_start_time
        message = 'Finished epoch {}, duration {}:{}:{}'.format(epoch+1, epoch_length.seconds//3600, (epoch_length.seconds//60)%60, epoch_length.seconds%60)
        model.eval()

        validation_load_iter = iter(validation_loader)
        validation_loss = 0
        validation_metrics = defaultdict(int)
        for data in validation_load_iter:
            with torch.no_grad():
                validation_loss += model.forward(data).item()
            for key, value in model.get_metrics().items():
                validation_metrics[key] += value.item()
        message += ', validation_loss - {}'.format(validation_loss / len(validation_loader.dataset))
        for key, value in model.get_metrics().items():
                message += ', validation_{} - {}'.format(key, value / len(validation_loader.dataset))
        
        print(message)
 

def _topological_loop(theta, batch_sizes):
    new = theta.new

    # batch_sizes is from packed_sequence, i.e. each element is the 'stride' within
    # the sequence data till the next element of the same batch
    B = batch_sizes[0].item()
    # So this, T, is the number of chunks in the packed sequence
    T = len(batch_sizes)
    # Theta itself is the packed sequence, so we have L as the flattened sequence length
    # S is the number of alignable states, note that this will be padded with infs, as the number of target states
    # varies across a batch
    L, S = theta.size()

    # Q stores history of gradients across sequence/batch
    Q = new(L, S+1).fill_(np.float('inf'))
    # V stores the history of potentials across the sequence
    # V[i, j] gives the cost of being in state j at time (i % batch_size) (i.e. this is a packed sequence). We pad with
    # an additional B elements to give an initial state of zero
    # Initialise with inf as some transitions at the start aren't valid - we begin in the upper left corner and not
    # all nodes are reachable from this parent node
    V = new(L + B, S+1).fill_(np.float('inf'))
    # i.e. we force position 0 for each batch (so a slice of elements of size B at the start of B) to be zero
    V[:B, 0] = 0

    left = B
    prev_length = B

    # For each step along sequence
    for t in range(1, T+1):
        # Handling the end case, which is just to put final result into Vt and Qt
        if t == T:
            cur_length = 0
        else:
            # cur_length means step to the next element in batch/size of chunk
            cur_length = batch_sizes[t]
        right = left + cur_length
        prev_left = left - prev_length
        # We cut at prev_right + cur_length - prev_length, to cut off any trailing batches
        # which we are no longer considering.
        prev_cut = right - prev_length
        len_term = prev_length - cur_length

        if cur_length != 0:
            # This is the dynamic part, the softmin of the total cost to get to each parent node
            v_prev, Q[left-B:right-B, :-1] = \
                torch.cat((V[prev_left:prev_cut, :-1, None], V[prev_left:prev_cut, 1:, None]), dim=-1).min(dim=-1)
            # Remember theta is a packed sequence, so this is a cut across batches at a sequence index
            # So the slice of theta is the potential of transitioning from si-1 to si
            V[left:right, 1:] = (theta[left-B:right-B, :] + v_prev)

        left = right
        prev_length = cur_length

    return V, Q
