import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
from e3nn.o3 import spherical_harmonics


def get_O3_attr(edge_index, pos, attr_irreps):
    """ Creates spherical harmonic edge attributes and node attributes for the SEGNN """
    rel_pos = pos[edge_index[0]] - pos[edge_index[1]]  # pos_j - pos_i (note in edge_index stores tuples like (j,i))
    edge_dist = rel_pos.pow(2).sum(-1, keepdims=True)
    edge_attr = spherical_harmonics(attr_irreps, rel_pos, normalize=True,
                                    normalization='component')  # Unnormalised for now
    node_attr = scatter(edge_attr, edge_index[1], dim=0, reduce="mean")
    return edge_attr, node_attr, edge_dist

def get_unique_atom_types(dataset):
    """
    Extract and return all unique atom types from the graphs in a dataset.

    Args:
        dataset (Dataset): A PyTorch Geometric dataset where each graph contains `x` as atom types.

    Returns:
        list: A sorted list of unique atom types found in the dataset.
    """
    unique_atom_types = set()

    for data in dataset:
        atom_types = data.x.view(-1).tolist()  # Flatten to 1D
        unique_atom_types.update(atom_types)

    return sorted(list(unique_atom_types))  # Return a sorted list for consistent indexing


def one_hot_encode_atom_types(atom_types, unique_atom_types):
    """
    Perform one-hot encoding for a list of atom types.

    Args:
        atom_types (Tensor): A tensor containing atom types (integers).
        unique_atom_types (list): A list of unique atom types.

    Returns:
        Tensor: One-hot encoded tensor of shape (num_atoms, len(unique_atom_types)).
    """
    num_types = len(unique_atom_types)
    atom_indices = torch.tensor([unique_atom_types.index(atom) for atom in atom_types])
    one_hot_encoded = torch.tensor(F.one_hot(atom_indices, num_classes=num_types), dtype=torch.float64) #.float()
    return one_hot_encoded

def processing_data(data, unique_atom_types, radius, attr_irreps):
    data.edge_index = radius_graph(data.pos, r=radius, batch=data.batch, loop=False)
    edge_attr, node_attr, edge_dist = get_O3_attr(data.edge_index, data.pos, attr_irreps)
    data.edge_attr, data.node_attr, data.edge_dist = edge_attr.float(), node_attr.float(), edge_dist.float()
    data.x = one_hot_encode_atom_types(data.x, unique_atom_types).float()
    data.additional_message_features = (data.edge_dist.pow(2)).float()
    return data
