import math
import torch.distributed as dist
from torch.utils.data import Sampler
import os
import torch
from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def make_dataloader(dataset, batch_size, num_workers, world_size=None, rank=None, train=True):
    """ Create (disributed) dataloader """

    if world_size is not None and world_size > 1:
        if train:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        else:
            sampler = DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)

        parallel_batch_size = int(batch_size/world_size)
        dataloader = DataLoader(dataset, batch_size=parallel_batch_size, shuffle=(sampler is None),
                                sampler=sampler, num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)

    return dataloader


def save_model(model, dir, id, gpu=""):
    """ Save a model """
    os.makedirs(dir, exist_ok=True)
    if gpu != "":
        gpu = "_" + str(gpu)

    torch.save(model.state_dict(), os.path.join(dir, id + gpu + ".pt"))


def load_model(model, dir, id, gpu=""):
    """ Load a state dict into a model """
    if gpu != "":
        gpu = "_" + str(gpu)
    state_dict = torch.load(os.path.join(dir, id + gpu + ".pt"))
    model.load_state_dict(state_dict)
    return model


class DistributedEvalSampler(Sampler):
    r"""
    Taken from https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py

    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default
    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.
    Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    Example::
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch

def generate_colors(num_colors):
    """
    Generate distinct colors for use in matplotlib plots.
    
    Parameters:
        num_colors (int): The number of distinct colors to generate.
    
    Returns:
        List of RGB tuples representing distinct colors.
    """
    # Use a color map from matplotlib to generate a range of colors
    color_map = plt.cm.get_cmap('tab10', num_colors) if num_colors <= 10 else plt.cm.get_cmap('hsv', num_colors)
    return [color_map(i) for i in range(num_colors)]

def visualize_and_save_lists(lists, lists_labels, 
                             filename="MSE_comparison_plot.png"):
    """
    Visualize two lists as line plots and save the figure.

    Parameters:
        list1 (list): The first list of values to plot.
        list2 (list): The second list of values to plot.
        filename (str): The filename to save the figure.
    """
    avg_list = []

    for list in lists:
        avg_list.append(np.mean(np.array(list)))

    # Create the plot
    plt.figure(figsize=(16, 12))
    color_list = generate_colors(len(lists))

    for index, list in enumerate(lists):
        plt.plot(list, label=lists_labels[index] + ', mean: ' + str(avg_list[index]), marker='o', 
                 linestyle='-', color=color_list[index])

    # Add labels, title, and legend
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Comparison of Two Lists")
    plt.legend()

    # Save the figure
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
