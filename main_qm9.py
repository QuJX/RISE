import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch_geometric.explain import Explainer
from qm9_SEGNN.utils import make_dataloader
from QM9_Explainer_setting import model_setting
from utils import save_dict_to_csv

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Dictionary mapping target indices to property names
qm9_target_dict = {
    0: 'mu', 1: 'alpha', 2: 'homo', 3: 'lumo', 4: 'gap',
    5: 'electronic_spatial_extent', 6: 'zpve', 7: 'energy_U0',
    8: 'energy_U', 9: 'enthalpy_H', 10: 'free_energy', 11: 'heat_capacity',
}

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--explained_model_name', type=str, default='SchNet', help='Name of the model to explain')
    parser.add_argument('--target_attr', type=int, default=3, help='Target attribute index')
    parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--budget', type=float, default=0.15, help='Budget for explanation')
    parser.add_argument('--checkpoint', type=str, default='saved_models/', help='Model checkpoint path')
    return parser.parse_args()

def get_explainer(model_name, setting, epochs):
    """Initialize the appropriate explainer based on the model name."""
    if model_name == 'SchNet':
        from RISE.gnn_explainer_Schnet import TDGNNExplainer
    elif model_name == 'DimeNet':
        from RISE.gnn_explainer_DimeNet import TDGNNExplainer
    elif model_name == 'SEGNN':
        from RISE.gnn_explainer_SEGNN import TDGNNExplainer
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return TDGNNExplainer(epochs=epochs, Explainer_setting=setting).to(setting.device)

def initialize_explanation_tracking():
    """Initialize dictionaries to track explanation results."""
    result_dict = {
        "Explain_loss": [],
        "Explain_train_loss": [],
        "Explain_result": [],
        "Explain_Edge_Keep_Ratio": [],
    }
    return result_dict

def main():
    args = parse_arguments()
    setting = model_setting(args.explained_model_name, args.target_attr, args.budget, args.checkpoint)
    device = setting.device
    
    Exp_algorithm = get_explainer(args.explained_model_name, setting, args.epoch)
    loader = make_dataloader(setting.datasets, batch_size=1, num_workers=1, world_size=1, rank=0, train=False) if args.explained_model_name == 'SEGNN' else DataLoader(setting.datasets[-1], batch_size=1, shuffle=False)

    explainer = Explainer(
        model=setting.model,
        algorithm=Exp_algorithm,
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(mode='regression', task_level='graph', return_type='raw'),
    )

    result_dict = initialize_explanation_tracking()

    for i, graph in enumerate(tqdm(loader, desc=f'{qm9_target_dict[args.target_attr]} on {args.explained_model_name}')):
        graph = graph.to(device)
        if args.explained_model_name != 'SEGNN':
            explanation = explainer(graph.z, graph.pos, target=None)
        else:
            target_mean, target_mad = loader.dataset.calc_stats()
            graph.mean_mad = torch.Tensor([target_mean, target_mad]).to(device)
            explanation = explainer(graph, target=None)
        with torch.no_grad():
            if args.explained_model_name != 'SEGNN':
                pred, loss, preserved_ratio = Exp_algorithm.masked_prediction(graph.z, graph.pos, graph.y[0, args.target_attr], None, edge_mask=explanation.edge_mask)
            else:
                pred, loss, preserved_ratio = Exp_algorithm.masked_prediction(graph, None, edge_mask=explanation.edge_mask)
            
            result_dict['Explain_loss'].append(float(loss.detach().cpu().numpy()))
            result_dict['Explain_train_loss'].append(float(Exp_algorithm.train_loss))
            result_dict['Explain_result'].append(float(pred.detach().cpu()))
            result_dict['Explain_Edge_Keep_Ratio'].append(round(preserved_ratio, 4))
                
        if i > 998 and args.explained_model_name == 'SEGNN':
            break

    save_dict_to_csv(result_dict, 'qm9_exp', f"{qm9_target_dict[args.target_attr]}_Epoch={args.epoch}_Budget={args.budget}.csv" )

if __name__ == "__main__":
    main()
