import torch
from torch_geometric.nn import SchNet, DimeNet
import os.path as osp
import torch.nn.functional as F
from SEGNN_model.segnn.segnn import SEGNN
from torch_geometric.datasets import QM9
from qm9_SEGNN.dataset import QM9_SEGNN
from e3nn.o3 import Irreps
from SEGNN_model.balanced_irreps import WeightBalancedIrreps
import warnings
import os

warnings.filterwarnings("ignore")

qm9_target_dict = {
    0: 'mu',
    1: 'alpha',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}

class model_setting():
    def __init__(self, explained_model_name = 'SchNet', target_attr = 0, budget = 0.5, checkpoint = 'Saved_models/segnn_qm9_mu_r=5_layer=7.pt'):
        self.path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
        self.dataset = QM9(self.path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_attr = target_attr
        self.Budget = budget
        self.explained_model_name = explained_model_name
        if self.explained_model_name == 'SchNet':
            self.model, self.datasets = SchNet.from_qm9_pretrained(self.path, self.dataset, self.target_attr)
            self.radius = self.model.cutoff
            self.dataset = self.dataset = [-1]
        elif self.explained_model_name == 'DimeNet':
            self.model, self.datasets = DimeNet.from_qm9_pretrained(self.path, self.dataset, self.target_attr)
            self.radius = self.model.cutoff
            self.dataset = self.dataset = [-1]
        elif self.explained_model_name == 'SEGNN':

            self.radius = 5
            num_layers = 7
            input_irreps = Irreps("5x0e")
            output_irreps = Irreps("1x0e")
            edge_attr_irreps = Irreps.spherical_harmonics(3)
            node_attr_irreps = Irreps.spherical_harmonics(3)
            additional_message_irreps = Irreps("1x0e")
            hidden_irreps = WeightBalancedIrreps(Irreps("{}x0e".format(128)), 
                                                 node_attr_irreps, sh=True, lmax=2)
            self.model = SEGNN(input_irreps,
                          hidden_irreps,
                          output_irreps,
                          edge_attr_irreps,
                          node_attr_irreps,
                          num_layers=num_layers,
                          norm="batch",
                          pool="avg",
                          task="graph",
                          additional_message_irreps=additional_message_irreps,
                          cutoff=self.radius)
            model_dict = torch.load(checkpoint, map_location=torch.device('cuda'))
            self.model.load_state_dict(model_dict)
            self.datasets = QM9_SEGNN('data', qm9_target_dict[target_attr], self.radius, "test", 3,
                                     feature_type='one_hot')
            
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        
