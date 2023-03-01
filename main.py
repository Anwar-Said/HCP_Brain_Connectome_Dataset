import networkx as nx
from utils import *
from torch_geometric.utils import to_networkx

source_dir = "data/"
dataset_name = 'gender_brain_dataset_5'
threshold = 5
dataset = Brain_Connectome_Dataset("data/",dataset_name,source_dir, threshold)
print("dataset preprocessing successfully completed!")
def compute_dataset_stats(dataset):
    num_classes = dataset.num_classes
    num_features = dataset.num_features
    nodes, edges,labels = [],[],[]
    for d in dataset:
        nodes.append(d.num_nodes)
        edges.append(d.num_edges)
        labels.append(d.y.item())
    print("dataset stats")
    print("total instances:", len(nodes))
    print("num_classes:", num_classes)
    print("num_features:", num_features)
    print("male female ratio:", np.sum(labels), len(labels)-np.sum(labels))
    print("average number of nodes:", np.sum(nodes)/len(nodes))
    print("average edges:", np.sum(edges)/len(edges))
    print("min and max edges:",np.min(edges), np.max(edges))
compute_dataset_stats(dataset)




