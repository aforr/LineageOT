# Converts the Table S6 from Packer et al. 2019 to a NetworkX DiGraph of the reference C. elegans lineage tree

import pandas as pd
import networkx as nx
import pickle
import numpy as np


data_path = "data/" # assuming script is runn from c_elegans folder
supplement_s6_file = "table_s6.csv"

lineage_table = pd.read_csv(data_path + supplement_s6_file)


tree = nx.DiGraph()

# Handling the root separately because it has no parent
tree.add_node('root')
tree.nodes['root']['name'] = 'P0'
# time is division time
tree.nodes['root']['time'] = 1
tree.nodes['root']['time_to_parent'] = np.nan
tree.nodes['root']['data'] = lineage_table.loc[lineage_table['cell'] == 'P0']

def add_node(node_id, parent_id, data_table = lineage_table, t = tree):
    if node_id in tree.nodes or node_id == 'P0':
        print("Node '" + node_id + "' already in tree. No new node added.")
        return
    data = data_table.loc[data_table['cell'] == node_id]
    t.add_node(node_id)
    t.add_edge(parent_id, node_id)
        
    t.nodes[parent_id]['time'] = data['birth_time'].iloc[0]
    t.nodes[parent_id]['time_to_parent'] = (t.nodes[parent_id]['time']
                                            - t.nodes[parent_id]['data']['birth_time'].iloc[0])
    
    t.nodes[node_id]['name'] = node_id
    # arbitrary large number for cells that don't divide
    # (not np.inf because that makes the precision matrix singular in LineageOT)
    t.nodes[node_id]['time'] = 2000
    t.nodes[node_id]['time_to_parent'] = 2000 - t.nodes[parent_id]['time']
    t.nodes[node_id]['data'] = data
    return



# Next handling the nodes in the table where the name doesn't indicate the parent
add_node('AB', 'root', t = tree)
add_node('P1', 'root', t = tree)

add_node('EMS', 'P1', t = tree)
add_node('P2', 'P1', t = tree)

add_node('E', 'EMS', t = tree)
add_node('MS', 'EMS', t = tree)

add_node('P3', 'P2', t = tree)
add_node('C', 'P2', t = tree)

add_node('P4', 'P3', t = tree)
add_node('D', 'P3', t = tree)

add_node('Z2', 'P4', t = tree)
add_node('Z3', 'P4', t = tree)


# Now adding all nodes from table
for i, cell in zip(range(lineage_table.shape[0]), lineage_table['cell']):
    parent = cell[:-1]
    add_node(cell, parent, t = tree)
    


with open(data_path + "packer_pickle_lineage_tree.p", 'wb') as file:
    pickle.dump(tree, file)
