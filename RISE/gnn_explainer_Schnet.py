from math import sqrt
from typing import Optional, Union
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks
from torch_geometric.explain.config import MaskType, ModelMode
import os.path as osp
import torch.nn.functional as F
class TDGNNExplainer(ExplainerAlgorithm):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and node features that play a crucial role in the predictions
    made by a GNN.

    .. note::

        For an example of using :class:`GNNExplainer`, see
        `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_,
        `examples/explain/gnn_explainer_ba_shapes.py <https://github.com/
        pyg-team/pytorch_geometric/blob/master/examples/
        explain/gnn_explainer_ba_shapes.py>`_, and `examples/explain/
        gnn_explainer_link_pred.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py>`_.

    .. note::

        The :obj:`edge_size` coefficient is multiplied by the number of nodes
        in the explanation at every iteration, and the resulting value is added
        to the loss as a regularization term, with the goal of producing
        compact explanations.
        A higher value will push the algorithm towards explanations with less
        elements.
        Consider adjusting the :obj:`edge_size` coefficient according to the
        average node degree in the dataset, especially if this value is bigger
        than in the datasets used in the original paper.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """
    coeffs = {
        'edge_size': 0.005,  ## budget, penalize sum(edge masks)
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,    ## edge entropy for sparcity default 1
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.001, Explainer_setting = None, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
        self.explained_model_name = Explainer_setting.explained_model_name
        self.Explained_model = Explainer_setting.model
        self.device = Explainer_setting.device
        self.Budget = Explainer_setting.Budget
        self.train_loss = 0

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        pos: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(pos, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")
        self._train(x, pos, target=target, index=index, **kwargs)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )
        self._clean_model(self.Explained_model)

        return Explanation(node_mask=node_mask, edge_mask=edge_mask, edges = self.edge_index)

    def supports(self) -> bool:
        return True

    def _train(
        self,
        x: Tensor,
        pos: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):  
        batch = kwargs['batch'].to(self.device) if 'batch' in kwargs else None
        batch = torch.zeros(x.shape[0], dtype=int).to(self.device) if batch is None else batch

        edge_index, edge_weight = self.Explained_model.interaction_graph(pos, batch)
        self.edge_index = edge_index
        self._initialize_masks(x, edge_index)

        parameters = []

        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        optimizer.zero_grad()
        self.Explained_model.eval()
        for i in range(self.epochs):

            h = self.Explained_model.embedding(x)      #### h: node embedding
            act = nn.Sigmoid()
            softmax= nn.Softmax(0)
            Node_radius = softmax(self.edge_mask) * self.edge_mask.shape[0] * self.Budget
            Edge_related_weight = edge_weight / self.Explained_model.cutoff
            start_nodes = edge_index[0]
            edge_mask = act(100 * (Node_radius[start_nodes] - Edge_related_weight))
            edge_attr = self.Explained_model.distance_expansion(edge_weight)   #### edge features
            edge_attr = edge_mask.unsqueeze(1) * edge_attr

            for interaction in self.Explained_model.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            h = self.Explained_model.lin1(h)
            h = self.Explained_model.act(h)
            h = self.Explained_model.lin2(h)

            if self.Explained_model.dipole:
                # Get center of mass.
                mass = self.Explained_model.atomic_mass[x].view(-1, 1)
                M = self.Explained_model.sum_aggr(mass, batch, dim=0)
                c = self.Explained_model.sum_aggr(mass * pos, batch, dim=0) / M
                h = h * (pos - c.index_select(0, batch))

            if not self.Explained_model.dipole and self.Explained_model.mean is not None and self.Explained_model.std is not None:
                h = h * self.Explained_model.std + self.Explained_model.mean

            if not self.Explained_model.dipole and self.Explained_model.atomref is not None:
                h = h + self.Explained_model.atomref(x)

            out = self.Explained_model.readout(h, batch, dim=0)

            if self.Explained_model.dipole:
                out = torch.norm(out, dim=-1, keepdim=True)

            if self.Explained_model.scale is not None:
                out = self.Explained_model.scale * out
                
            y_hat, y = out, target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            if y_hat.shape != y.shape:
                y = y.view(y_hat.shape)
            loss = self._loss(y_hat, y)
                
            loss.backward(retain_graph=True)
            optimizer.step()

            if i == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError("Could not compute gradients for node "
                                     "features. Please make sure that node "
                                     "features are used inside the model or "
                                     "disable it via `node_mask_type=None`.")
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                if self.edge_mask.grad is None:
                    raise ValueError("Could not compute gradients for edges. "
                                     "Please make sure that edges are used "
                                     "via message passing inside the model or "
                                     "disable it via `edge_mask_type=None`.")
                self.hard_edge_mask = self.edge_mask.grad != 0.0
        
        self.train_loss = loss.detach().cpu().numpy()
    
    def masked_prediction(self,
        x: Tensor,
        pos: Tensor,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        edge_mask = None,
        node_mask = None,
        **kwargs):
        batch = kwargs['batch'].to(self.device) if 'batch' in kwargs else None
        batch = torch.zeros(x.shape[0], dtype=int).to(self.device) if batch is None else batch
        self.Explained_model.eval()
        
        edge_index, edge_weight = self.Explained_model.interaction_graph(pos, batch)
        self.edge_index = edge_index
        h = self.Explained_model.embedding(x)      
        
        act = nn.Sigmoid()
        softmax= nn.Softmax(0)
        Node_radius = softmax(edge_mask) * edge_mask.shape[0] * self.Budget
        Edge_related_weight = edge_weight / self.Explained_model.cutoff
        start_nodes = edge_index[0]
        Preserved_Edge = Node_radius[start_nodes] - Edge_related_weight
        edge_num = edge_weight.shape[0]
        edge_index = edge_index[:, Preserved_Edge > 0]
        edge_weight = edge_weight[Preserved_Edge > 0]
        preserved_ratio = edge_weight.shape[0] / edge_num

        edge_attr = self.Explained_model.distance_expansion(edge_weight)   #### edge features

        for interaction in self.Explained_model.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.Explained_model.lin1(h)
        h = self.Explained_model.act(h)
        h = self.Explained_model.lin2(h)

        if self.Explained_model.dipole:
            # Get center of mass.
            mass = self.Explained_model.atomic_mass[x].view(-1, 1)
            M = self.Explained_model.sum_aggr(mass, batch, dim=0)
            c = self.Explained_model.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not self.Explained_model.dipole and self.Explained_model.mean is not None and self.Explained_model.std is not None:
            h = h * self.Explained_model.std + self.Explained_model.mean

        if not self.Explained_model.dipole and self.Explained_model.atomref is not None:
            h = h + self.Explained_model.atomref(x)

        out = self.Explained_model.readout(h, batch, dim=0)

        if self.Explained_model.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.Explained_model.scale is not None:
            out = self.Explained_model.scale * out
        y_hat, y = out, target

        if index is not None:
            y_hat, y = y_hat[index], y[index]
        
        loss = F.l1_loss(y_hat.squeeze(0), y.unsqueeze(0))
        
        return y_hat, loss, preserved_ratio
    
    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        #(M, N, F), E = x.size(), edge_index.size(1)
        (N,), E = x.size(), edge_index.size(1)
        F = 1
        
        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            assert False

        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            self.edge_mask = Parameter(torch.ones(N, device=device))

        else:
            assert False

    def _loss_binary_classification(self,y_hat, y):
        ce = torch.nn.CrossEntropyLoss(reduction='none')
        return ce(y_hat, y)
    
    def _loss_regression(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return F.l1_loss(y_hat, y) 
    
    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            assert False
        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
