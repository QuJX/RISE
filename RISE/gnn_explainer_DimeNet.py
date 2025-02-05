from math import sqrt
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel
from torch_geometric.nn.pool import radius_graph
import os.path as osp
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn.models.dimenet import triplets

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
    """
    geo dimenet 0.005 0
    """

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
        self._train(model, x, pos, target=target, index=index, **kwargs)

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
        self._clean_model(model)

        return Explanation(node_mask=node_mask, edge_mask=edge_mask, edges = self.edge_index)

    def supports(self) -> bool:
        return True

    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        pos: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):  
        # clear_masks(model)
        z = x
        batch = kwargs['batch'].to(self.device) if 'batch' in kwargs else None
        batch = torch.zeros(x.shape[0], dtype=int).to(self.device) if batch is None else batch

        edge_index = radius_graph(pos, r=model.cutoff, batch=batch,
                                  max_num_neighbors=model.max_num_neighbors)
        self.edge_index = edge_index
        self._initialize_masks(x, edge_index)

        parameters = []

        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            # set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        optimizer.zero_grad()

        for epoch in range(self.epochs):
            
            i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
                edge_index, num_nodes=z.size(0))
            
            dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
            pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki, dim=1).norm(dim=-1)
            angle = torch.atan2(b, a)

            act = nn.Sigmoid()
            softmax= nn.Softmax(0)
            Node_radius = softmax(self.edge_mask) * self.edge_mask.shape[0] * self.Budget
            Edge_related_weight = dist / self.Explained_model.cutoff
            start_nodes = edge_index[0]
            edge_mask = act(100 * (Node_radius[start_nodes] - Edge_related_weight))

            rbf = model.rbf(dist) 
            sbf = model.sbf(dist, angle, idx_kj)

            x = model.emb(z, rbf, i, j)
            x = x * edge_mask.unsqueeze(1)

            P = model.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

            for interaction_block, output_block in zip(model.interaction_blocks,
                                                   model.output_blocks[1:]):
                x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
                P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

            if batch is None:
                out = P.sum(dim=0)
            else:
                out = scatter(P, batch, dim=0, reduce='sum')
            
            out = out.view(-1)

            y_hat, y = out, target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            if y.shape != y_hat.shape:
                y = y.squeeze(1)

            loss = self._loss(y_hat, y)
                
            loss.backward(retain_graph=True)
            optimizer.step()

            if epoch == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError("Could not compute gradients for node "
                                     "features. Please make sure that node "
                                     "features are used inside the model or "
                                     "disable it via `node_mask_type=None`.")
                self.hard_node_mask = self.node_mask.grad != 0.0
            if epoch == 0 and self.edge_mask is not None:
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
        
        edge_index = radius_graph(pos, r=self.Explained_model.cutoff, batch=batch,
                                  max_num_neighbors=self.Explained_model.max_num_neighbors)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
                edge_index, num_nodes=x.size(0))
        
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        act = nn.Sigmoid()
        softmax= nn.Softmax(0)
        Node_radius = softmax(edge_mask) * edge_mask.shape[0] * self.Budget
        Edge_related_weight = dist / self.Explained_model.cutoff
        start_nodes = edge_index[0]
        Preserved_Edge = Node_radius[start_nodes] - Edge_related_weight
        edge_num = edge_index.shape[1]
        edge_index = edge_index[:, Preserved_Edge > 0]
        dist = dist[Preserved_Edge > 0]
        preserved_ratio = edge_index.shape[1] / edge_num

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
                edge_index, num_nodes=x.size(0))


        if self.explained_model_name == 'DimeNet++':
                pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
                a = (pos_ij * pos_jk).sum(dim=-1)
                b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        elif self.explained_model_name == 'DimeNet':
            pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki, dim=1).norm(dim=-1)
        
        angle = torch.atan2(b, a)

        rbf = self.Explained_model.rbf(dist)
        sbf = self.Explained_model.sbf(dist, angle, idx_kj)

        x = self.Explained_model.emb(x, rbf, i, j)
        P = self.Explained_model.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        for interaction_block, output_block in zip(self.Explained_model.interaction_blocks,
                                            self.Explained_model.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        if batch is None:
            out = P.sum(dim=0)
        else:
            out = scatter(P, batch, dim=0, reduce='sum')
        
        out = out.view(-1)

        y_hat, y = out, target

        if index is not None:
            y_hat, y = y_hat[index], y[index]

        if y.shape != y_hat.shape:
            y = y.view(y_hat.shape)

        loss = F.l1_loss(y_hat, y)
        
        return y_hat, loss, preserved_ratio
    
    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        (N,), E = x.size(), edge_index.size(0)
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
            self.edge_mask = Parameter(torch.randn(N, device=device))

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


class GNNExplainer_:
    r"""Deprecated version for :class:`GNNExplainer`."""

    coeffs = TDGNNExplainer.coeffs

    conversion_node_mask_type = {
        'feature': 'common_attributes',
        'individual_feature': 'attributes',
        'scalar': 'object',
    }

    conversion_return_type = {
        'log_prob': 'log_probs',
        'prob': 'probs',
        'raw': 'raw',
        'regression': 'raw',
    }

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        return_type: str = 'log_prob',
        feat_mask_type: str = 'feature',
        allow_edge_mask: bool = True,
        **kwargs,
    ):
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']

        explainer_config = ExplainerConfig(
            explanation_type='model',
            node_mask_type=self.conversion_node_mask_type[feat_mask_type],
            edge_mask_type=MaskType.object if allow_edge_mask else None,
        )
        model_config = ModelConfig(
            mode='regression'
            if return_type == 'regression' else 'multiclass_classification',
            task_level=ModelTaskLevel.node,
            return_type=self.conversion_return_type[return_type],
        )

        self.model = model
        self._explainer = GNNExplainer(epochs=epochs, lr=lr, **kwargs)
        self._explainer.connect(explainer_config, model_config)

    @torch.no_grad()
    def get_initial_prediction(self, *args, **kwargs) -> Tensor:

        training = self.model.training
        self.model.eval()

        out = self.model(*args, **kwargs)
        if (self._explainer.model_config.mode ==
                ModelMode.multiclass_classification):
            out = out.argmax(dim=-1)

        self.model.train(training)

        return out

    def explain_graph(
        self,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        self._explainer.model_config.task_level = ModelTaskLevel.graph

        explanation = self._explainer(
            self.model,
            x,
            edge_index,
            target=self.get_initial_prediction(x, edge_index, **kwargs),
            **kwargs,
        )
        return self._convert_output(explanation, edge_index)

    def explain_node(
        self,
        node_idx: int,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        self._explainer.model_config.task_level = ModelTaskLevel.node
        explanation = self._explainer(
            self.model,
            x,
            edge_index,
            target=self.get_initial_prediction(x, edge_index, **kwargs),
            index=node_idx,
            **kwargs,
        )
        return self._convert_output(explanation, edge_index, index=node_idx,
                                    x=x)

    def _convert_output(self, explanation, edge_index, index=None, x=None):
        node_mask = explanation.get('node_mask')
        edge_mask = explanation.get('edge_mask')

        if node_mask is not None:
            node_mask_type = self._explainer.explainer_config.node_mask_type
            if node_mask_type in {MaskType.object, MaskType.common_attributes}:
                node_mask = node_mask.view(-1)

        if edge_mask is None:
            if index is not None:
                _, edge_mask = self._explainer._get_hard_masks(
                    self.model, index, edge_index, num_nodes=x.size(0))
                edge_mask = edge_mask.to(x.dtype)
            else:
                edge_mask = torch.ones(edge_index.size(1),
                                       device=edge_index.device)

        return node_mask, edge_mask
