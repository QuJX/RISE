import torch
from torch_geometric.nn import SchNet, DimeNet
import os.path as osp
from SEGNN_model.segnn.segnn import SEGNN
from geo_SEGNN.dataset import get_unique_atom_types
from e3nn.o3 import Irreps
from SEGNN_model.balanced_irreps import WeightBalancedIrreps
import warnings
import os

warnings.filterwarnings("ignore")


class model_setting():
    def __init__(self, explained_model_name = 'SchNet', budget = 0.5):
        self.path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'geo', 'Geo_data.pt')
        self.datasets = torch.load(self.path)
        self.unique_atom_types = get_unique_atom_types(self.datasets)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Budget = budget
        self.explained_model_name = explained_model_name
        self.radius = 5
        if self.explained_model_name == 'SchNet':
            self.model = SchNet(
                hidden_channels=128,
                num_filters=128,
                num_interactions=6,
                num_gaussians=50,
                cutoff=5,
                max_num_neighbors=32).to(self.device)
            self.model.load_state_dict(torch.load('Saved_models/SchNet_Geo.pth', map_location=self.device))
            self.radius = self.model.cutoff
        elif self.explained_model_name == 'DimeNet':
            self.model = DimeNet(
                hidden_channels=128,
                out_channels=1,
                num_blocks=6,
                num_bilinear=8,
                num_spherical=7,
                num_radial=6,
                cutoff=5.0,
                envelope_exponent=5,
                num_before_skip=1,
                num_after_skip=2,
                num_output_layers=3,
            ).to(self.device)
            self.model.load_state_dict(torch.load('Saved_models/DimeNet_Geo.pth', map_location=self.device))
            self.radius = self.model.cutoff
        elif self.explained_model_name == 'SEGNN':

            self.radius = 5
            num_layers = 7
            input_irreps = Irreps("16x0e")
            output_irreps = Irreps("1x0e")
            edge_attr_irreps = Irreps.spherical_harmonics(3)
            node_attr_irreps = Irreps.spherical_harmonics(3)
            self.attr_irreps = Irreps.spherical_harmonics(3)
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
                          additional_message_irreps=additional_message_irreps)
            
            model_dict = torch.load(os.path.join('Saved_models', f"SEGNN_geo.pt"), 
                                                                map_location=torch.device('cuda'))
            
            self.model.load_state_dict(model_dict)
        
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        
