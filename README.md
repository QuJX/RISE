# <p align=center> [In Submission] Radius of Influence based Subgraph Extraction for 3D Molecular Graph Explanation</p>

<div align="center">

[![Paper](https://img.shields.io/badge/PDF-Paper-red.svg)](https://github.com/QuJX/RISE)
[![Web](https://img.shields.io/badge/RISE-Web-blue.svg)](https://github.com/QuJX/RISE)
[![Code](https://img.shields.io/badge/RISE-Code-orange.svg)](https://github.com/QuJX/RISE)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FQuJX%2FRISE&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>

---

>**RISE: Radius of Influence based Subgraph Extraction for 3D Molecular Graph Explanation** <br>
>[Jingxiang Qu](https://tom-q.netlify.app/)<sup># </sup>, [Wenhan Gao](https://wenhangao21.github.io/)<sup># </sup>, [Jiaxing Zhang](https://tabzhangjx.github.io/), [Xufeng Liu](https://xufliu.github.io/), [Hua Wei](https://www.public.asu.edu/~hwei27/index.html), [Haibin Ling](https://www3.cs.stonybrook.edu/~hling),  [Yi Liu](https://jacoblau0513.github.io/)<sup>* </sup> <br>
>(* Corresponding Author) <br>
>(# Equal Contribution) <br> 
>In Submission <br>

> **Introduction:** *3D Geometric Graph Neural Networks (GNNs) have emerged as transformative tools for modeling molecular data. Despite their predictive power, these models often suffer from limited interpretability, raising concerns for scientific applications that require reliable and transparent insights. While existing methods have primarily focused on explaining molecular substructures in 2D GNNs, the transition to 3D GNNs introduces unique challenges, such as handling the implicit dense edge structures created by a cutoff radius. To tackle this, we introduce a novel explanation method specifically designed for 3D GNNs, which localizes the explanation to the immediate neighborhood of each node within the 3D space. Each node is assigned an radius of influence, defining the localized region within which message passing captures spatial and structural interactions crucial for the model's predictions. This method leverages the spatial and geometric characteristics inherent in 3D graphs. By constraining the subgraph to a localized radius of influence, the approach not only enhances interpretability but also aligns with the physical and structural dependencies typical of 3D graph applications, such as molecular learning.*
<hr />

## Examples
![Chemical Structures](https://github.com/user-attachments/assets/6dd8eea3-f151-49a8-aa22-a6bd65bde0ae)
![radius_reduction](https://github.com/user-attachments/assets/b68a6cff-7d8b-4d65-84e9-78d18f1ad803)

## Environment
[Requirements.txt](requirements.txt)

## Running

Before running the experiment, please remove the line 250 in torch_geometric.explain.explainer. <br />
Because our method don't have to set any threshold to the edge_mask, and we don't need to validate the size of mask, which are kept the same as the original edge number. <br />

#### QM9
To run the QM9 experiments, adapt explained_model_name, target_attr, epoch, budget, and checkpoint. (If running on SchNet or DimeNet, the 'checkpoint' can be ignored.) <br />
If you want to test the explainer on SEGNN, you need to train the SEGNN and saved the model_stat_dict firstly.  <br />
The official version of SEGNN can be find <a href="https://github.com/RobDHess/Steerable-E3-GNN">here</a>. <br />
It is noted that the chemical properties in QM9 dataset are encoded follows 'target_attr' dict: <br />
{ <br />
    0: 'mu', 1: 'alpha', 2: 'homo', 3: 'lumo', 4: 'gap', <br />
    5: 'electronic_spatial_extent', 6: 'zpve', 7: 'energy_U0', <br />
    8: 'energy_U', 9: 'enthalpy_H', 10: 'free_energy', 11: 'heat_capacity', <br />
}.<br />

```bash
python main_qm9.py --explained_model_name=SchNet --target_attr=0 --epoch=200 --budget=0.5
```

#### GEOM
To run the GEOM experiments, adapt Explained_model_name, epoch, budget, and checkpoint. (Please train the backbone model on GEOM Dataset First.) <br />

```bash
python main_geom.py --explained_model_name=SchNet --epoch=200 --budget=0.5 --checkpoint='your_checkpoint_path.pt'
```

</div>
<p align="center"> 
  Visitor count<br>
  <img src="https://profile-counter.glitch.me/QuJX_RISE/count.svg" />
</p>
