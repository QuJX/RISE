import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch_geometric.explain import Explainer
from GEOM_Explainer_setting import model_setting
from geo_SEGNN.dataset import processing_data
from qm9_SEGNN.utils import make_dataloader
from utils import save_dict_to_csv

# Set random seed for reproducibility
torch.manual_seed(7)
np.random.seed(7)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--explained_model_name', type=str, default='SchNet', help='Model to explain')
    parser.add_argument('--epoch', type=int, default=1, help='Training epochs')
    parser.add_argument('--budget', type=float, default=0.4, help='Explanation budget')
    return parser.parse_args()

def get_explainer(model_name, setting, epochs):
    if model_name == 'SchNet':
        from RISE.gnn_explainer_Schnet import TDGNNExplainer
    elif model_name in ['DimeNet', 'DimeNet++']:
        from RISE.gnn_explainer_DimeNet import TDGNNExplainer
    elif model_name == 'SEGNN':
        from RISE.gnn_explainer_SEGNN import TDGNNExplainer
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return TDGNNExplainer(epochs=epochs, Explainer_setting=setting).to(setting.device)

def initialize_explanation_tracking():
    return {"Explain_loss": [], "Explain_train_loss": [], "Explain_result": [], "Explain_Edge_Keep_Ratio": [], "Explain_MAE": []}

def main():
    args = parse_arguments()
    setting = model_setting(args.explained_model_name, args.budget)
    device = setting.device
    
    Exp_algorithm = get_explainer(args.explained_model_name, setting, args.epoch)
    
    if args.explained_model_name == 'SEGNN':
        loader = make_dataloader(setting.datasets[50000:51000], batch_size=1, num_workers=1, world_size=1, rank=0, train=False)
    else:
        loader = DataLoader(setting.datasets[50000:51000], batch_size=1, shuffle=False)
    
    explainer = Explainer(
        model=setting.model,
        algorithm=Exp_algorithm,
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(mode='regression', task_level='graph', return_type='raw'),
    )
    
    result_dict = initialize_explanation_tracking()
    
    for i, graph in enumerate(tqdm(loader, desc=f'RISE GEO energy on {args.explained_model_name}')):
        graph = graph.to(device)
        if i > 2:
            break
        if args.explained_model_name == 'SEGNN':
            graph = processing_data(graph, unique_atom_types=setting.unique_atom_types, radius=setting.radius, attr_irreps=setting.attr_irreps)
            graph.mean_mad = torch.Tensor([0, 1]).to(device)
        
        explanation = explainer(graph if args.explained_model_name == 'SEGNN' else (graph.x, graph.pos.float()), target=graph.y)
        
        with torch.no_grad():
            pred, loss, preserved_ratio = Exp_algorithm.masked_prediction(graph if args.explained_model_name == 'SEGNN' else (graph.x, graph.pos.float()), None, edge_mask=explanation.edge_mask)
            
            result_dict['Explain_loss'].append(float(loss.detach().cpu().numpy()))
            result_dict['Explain_train_loss'].append(float(Exp_algorithm.train_loss))
            result_dict['Explain_result'].append(float(pred.detach().cpu()))
            result_dict['Explain_Edge_Keep_Ratio'].append(round(preserved_ratio, 4))
            result_dict['Explain_MAE'].append(abs(pred.detach().cpu().numpy() - graph.y.detach().cpu().numpy()))
    
    save_dict_to_csv(result_dict, 'geom_exp', f"RISE_GEO_Energy_Epoch={args.epoch}_Budget={args.budget}.csv")

if __name__ == "__main__":
    main()
