import itertools
import functools
from typing import Iterable, Union

import networkx as nx
import rustworkx as rx

import pennylane as qml
from pennylane.wires import Wires
from pennylane import numpy as nps
from pennylane.qaoa.cost import bit_driver

def cost_ds(graph: Union[nx.Graph, rx.PyGraph]):
    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph, got {type(graph).__name__}"
        )

    graph_nodes = graph.nodes()
    coeffs = []
    terms  = []
    for i in graph_nodes:
        coeffs +=[0.5,-0.5]
        terms  += [qml.Identity(i),qml.PauliZ(i)]
    return qml.Hamiltonian(coeffs,terms)



def mixer_ds(graph: Union[nx.Graph, rx.PyGraph]):
    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph object, got {type(graph).__name__}"
        )

    coeffs = []
    terms = []

    is_rx = isinstance(graph, rx.PyGraph)
    graph_nodes = graph.node_indexes() if is_rx else graph.nodes

    # In RX each node is assigned to an integer index starting from 0;
    # thus, we use the following lambda function to get node-values.
    get_nvalue = lambda i: graph.nodes()[i] if is_rx else i
    
    for u in graph_nodes:
        neighbours_u = list(graph.neighbors(u))
        neighbours_u +=[u]
        n_terms = [[qml.PauliX(u)]]
        n_coeffs = [[1]]
        for v in neighbours_u:
            neighbours_v = list(graph.neighbors(v))
            neighbours_v +=[v]
            neighbours_v.remove(u)
            n_terms_2 = []
            n_coeffs_2 = []
            for w in neighbours_v:
                n_terms_2 +=[qml.Identity(w),qml.PauliZ(w)]
                n_coeffs_2 +=[0.5,-0.5]
            n_terms +=[n_terms_2]
            n_coeffs +=[n_coeffs_2]

        if len(n_coeffs) != 1:
            final_terms = [qml.operation.Tensor(*list(m)).prune() for m in itertools.product(*n_terms)]
            final_coeffs = [functools.reduce(lambda x, y: x * y, list(m), 1)
                for m in itertools.product(*n_coeffs)
            ]

            coeffs.extend(final_coeffs)
            terms.extend(final_terms)

    return qml.Hamiltonian(coeffs, terms)