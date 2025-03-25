from graph_analysis import balanced_forman_curvature_sparse
import numpy as np
import networkx as nx
from plotly_visualize import visualize_graph

# # 1. Complete graph of size N=10
# G = nx.complete_graph(10)

# # # 2. Grid graph
# G = nx.grid_2d_graph(50, 50)
# G = nx.convert_node_labels_to_integers(G)


# G = nx.convert_node_labels_to_integers(G)
# C = balanced_forman_curvature(G)
# edge_curvatures = {}
# for u, v in G.edges():
#     # For undirected graphs, ensure consistent ordering
#     curvature = C[u, v] if (u, v) in G.edges else C[v, u]
#     edge_curvatures[(u, v)] = curvature
#
# visualize_graph(G, edge_curvatures, node_sizes=[15]*len(G), layout="graphviz", title="G", filename="graph.html")


# # 3. Tree-like graph with a central node
# G = nx.Graph()
# G.add_edges_from([
#     (0, 1), (0, 2), (0, 3),
#     (1, 4), (1, 5),
#     (2, 6),
#     (3, 7), (3, 8), (3, 9),
#     (9, 7), (9, 8), (9, 5),
#     (3, 1), (6, 10), (6, 11), (6, 12), (6, 13), (10, 11)
# ])


#
# C = balanced_forman_curvature(G)
#
# edge_curvatures = {}
# for u, v in G.edges():
#     # For undirected graphs, ensure consistent ordering
#     curvature = C[u, v] if (u, v) in G.edges else C[v, u]
#     edge_curvatures[(u, v)] = curvature
#
# visualize_graph(G, edge_curvatures, node_sizes=[15]*len(G), layout="graphviz", title='Tree-like Graph with Balanced Forman Ricci Curvature')


# # Create two complete graphs
# n1 = 10  # Number of nodes in the first complete graph
# n2 = 10  # Number of nodes in the second complete graph
# G1 = nx.complete_graph(n1)
# G2 = nx.complete_graph(n2)
#
# # Relabel nodes in G2 to avoid overlap with G1
# mapping = {i: i + n1 for i in G2.nodes()}
# G2 = nx.relabel_nodes(G2, mapping)
#
# # Combine the two graphs
# G = nx.compose(G1, G2)
#
# # Add more random edges between the two graphs
# num_additional_edges = 1  # Number of additional edges to add
# for _ in range(num_additional_edges):
#     node_from_G1 = random.choice(list(G1.nodes()))
#     node_from_G2 = random.choice(list(G2.nodes()))
#     G.add_edge(node_from_G1, node_from_G2)
#
# C = balanced_forman_curvature(G)
#
# # Assume G is your graph and C is the curvature matrix
#
# # Create a dictionary of edge curvatures
# edge_curvatures = {}
# for u, v in G.edges():
#     # For undirected graphs, ensure consistent ordering
#     curvature = C[u, v] if (u, v) in G.edges else C[v, u]
#     edge_curvatures[(u, v)] = curvature
#
# # Visualize the graph
# visualize_graph(G, edge_curvatures, node_sizes=[15]*len(G), layout="graphviz", title="Graph with Edge Curvatures", filename="graph.html")


# G = nx.Graph()

# a simple graph
# G.add_edges_from([
#     (0, 1), (0, 3), (0, 2), (0, 6), (0, 4),
#     (2, 5), (3, 5), (5, 1),
#     (4, 6), (6, 1)
# ])

# a cycle
# G.add_edges_from([
#     (0, 1), (1, 2), (2, 3), (3, 0)
# ])


# G.add_edges_from([
#     (0, 2), (0, 1), (0, 5),
#     (1, 2), (1, 3), (1, 4), (1, 5),
#     (2, 6), (3, 6), (4, 6), (5, 6),
#     (2, 7), (5, 7), (6, 7),
# ]
# )


# a bottleneck
# G.add_edges_from([
#     (0, 3), (1, 3), (2, 3),
#     (3, 4),
#     (4, 5), (4, 6), (4, 7)
# ])

# make bottleneck less bottlenecky
# G.add_edges_from([
#     (0, 5), (1, 5), (2, 7),
# ])

# a diamond (multiple paths)
# G.add_edges_from([
#     (0, 1), (0, 2), (0, 3), (0, 4),
#     (1, 5), (2, 5), (3, 5), (4, 5),
#     ])


import random
# Create two complete graphs
n1 = 10  # Number of nodes in the first complete graph
n2 = 10  # Number of nodes in the second complete graph
G1 = nx.complete_graph(n1)
G2 = nx.complete_graph(n2)

# Relabel nodes in G2 to avoid overlap with G1
mapping = {i: i + n1 for i in G2.nodes()}
G2 = nx.relabel_nodes(G2, mapping)

# Combine the two graphs
G = nx.compose(G1, G2)

# Add more random edges between the two graphs
num_additional_edges = 1  # Number of additional edges to add
for _ in range(num_additional_edges):
    node_from_G1 = random.choice(list(G1.nodes()))
    node_from_G2 = random.choice(list(G2.nodes()))
    G.add_edge(node_from_G1, node_from_G2)

C = balanced_forman_curvature_sparse(G)

edge_curvatures = {}
for u, v in G.edges():
    # For undirected graphs, ensure consistent ordering
    curvature = C[u, v] if (u, v) in G.edges else C[v, u]
    edge_curvatures[(u, v)] = curvature
    print(f"Curvature between nodes {u} and {v}: {curvature}")

cmaps = ["CMRmap", "RdBu_r", "RdYlGn", "Spectral_r", "bwr", "copper", "cividis",
         "gist_heat", "gist_earth", "managua", "inferno", "plasma", "vanimo_r"]
for cmap in cmaps:
    visualize_graph(G, edge_curvatures,
                    node_size=10,
                    cmap=cmap,
                    layout="graphviz",
                    title=f"Graph with Edge Curvatures {cmap}",
                    filename="graph.html")



