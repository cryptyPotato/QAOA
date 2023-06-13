import itertools
import functools
from typing import Iterable, Union

import networkx as nx
import rustworkx as rx

import pennylane as qml
from pennylane.wires import Wires
from pennylane import numpy as np
from pennylane.qaoa.cost import bit_driver

def cost_h(graph: Union[nx.Graph, rx.PyGraph],b: int):
    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph, got {type(graph).__name__}"
        )

    graph_nodes = graph.nodes()
    cost_h = bit_driver(graph_nodes, 0)
    cost_h.grouping_indices = [list(range(len(cost_h.ops)))]
    return cost_h
def bit_flip_mixer(graph: Union[nx.Graph, rx.PyGraph], b: int):
    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph object, got {type(graph).__name__}"
        )

    if b not in [0, 1]:
        raise ValueError(f"'b' must be either 0 or 1, got {b}")

    sign = 1 if b == 0 else -1

    coeffs = []
    terms = []

    is_rx = isinstance(graph, rx.PyGraph)
    graph_nodes = graph.node_indexes() if is_rx else graph.nodes
    # In RX each node is assigned to an integer index starting from 0;
    # thus, we use the following lambda function to get node-values.
    get_nvalue = lambda i: graph.nodes()[i] if is_rx else i
    for i in graph_nodes:
        neighbours = sorted(graph.neighbors(i)) if is_rx else list(graph.neighbors(i))
        degree = len(neighbours)

        n_terms = [[qml.PauliX(get_nvalue(i))]] + [
            [qml.Identity(get_nvalue(n)), qml.PauliZ(get_nvalue(n))] for n in neighbours
        ]
        n_coeffs = [[1, sign] for n in neighbours]

        final_terms = [qml.operation.Tensor(*list(m)).prune() for m in itertools.product(*n_terms)]
        final_coeffs = [
            (0.5**degree) * functools.reduce(lambda x, y: x * y, list(m), 1)
            for m in itertools.product(*n_coeffs)
        ]

        coeffs.extend(final_coeffs)
        terms.extend(final_terms)

    return qml.Hamiltonian(coeffs, terms)

def bit_flip_mixer_2(graph: Union[nx.Graph, rx.PyGraph], b: int):
    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph object, got {type(graph).__name__}"
        )

    if b not in [0, 1]:
        raise ValueError(f"'b' must be either 0 or 1, got {b}")

    sign = 1 if b == 0 else -1

    coeffs = []
    terms = []

    is_rx = isinstance(graph, rx.PyGraph)
    graph_nodes = graph.node_indexes() if is_rx else graph.nodes

    # In RX each node is assigned to an integer index starting from 0;
    # thus, we use the following lambda function to get node-values.
    get_nvalue = lambda i: graph.nodes()[i] if is_rx else i

    for u in graph_nodes:
        for v in graph_nodes:
            if u==v:
                i = u
                neighbours = sorted(graph.neighbors(i)) if is_rx else list(graph.neighbors(i))
                degree = len(neighbours)

                n_terms = [[qml.PauliX(get_nvalue(i))]] + [
                    [qml.Identity(get_nvalue(n)), qml.PauliZ(get_nvalue(n))] for n in neighbours
                ]
                n_coeffs = [[1, sign] for n in neighbours]

                final_terms = [qml.operation.Tensor(*list(m)).prune() for m in itertools.product(*n_terms)]
                final_coeffs = [
                    (0.5**degree) * functools.reduce(lambda x, y: x * y, list(m), 1)
                    for m in itertools.product(*n_coeffs)
                ]

                coeffs.extend(final_coeffs)
                terms.extend(final_terms)
            elif u < v and not graph.has_edge(u,v):
                neighbours_u = sorted(graph.neighbors(u)) if is_rx else list(graph.neighbors(u))
                neighbours_v = sorted(graph.neighbors(v)) if is_rx else list(graph.neighbors(v))
                neighbours   = sorted(list(set(neighbours_u) | set(neighbours_v)))
                degree       = len(neighbours)
                n_terms      = [[qml.PauliX(get_nvalue(u))@qml.PauliX(get_nvalue(v))]] +[[qml.Identity(get_nvalue(n)), qml.PauliZ(get_nvalue(n))] for n in neighbours]
                n_coeffs     = [[1, sign] for n in neighbours]
                final_terms = [qml.operation.Tensor(*list(m)).prune() for m in itertools.product(*n_terms)]
                final_coeffs = [
                    (0.5**degree) * functools.reduce(lambda x, y: x * y, list(m), 1)
                    for m in itertools.product(*n_coeffs)
                ]

                coeffs.extend(final_coeffs)
                terms.extend(final_terms)
    
    return qml.Hamiltonian(coeffs, terms)
                
def bit_flip_mixer_3(graph: Union[nx.Graph, rx.PyGraph], b: int):
    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph object, got {type(graph).__name__}"
        )

    if b not in [0, 1]:
        raise ValueError(f"'b' must be either 0 or 1, got {b}")

    sign = 1 if b == 0 else -1

    coeffs = []
    terms = []

    is_rx = isinstance(graph, rx.PyGraph)
    graph_nodes = graph.node_indexes() if is_rx else graph.nodes

    # In RX each node is assigned to an integer index starting from 0;
    # thus, we use the following lambda function to get node-values.
    get_nvalue = lambda i: graph.nodes()[i] if is_rx else i

    for u in graph_nodes:
        for v in graph_nodes:
            if u==v:
                i = u
                neighbours = sorted(graph.neighbors(i)) if is_rx else list(graph.neighbors(i))
                degree = len(neighbours)

                n_terms = [[qml.PauliX(get_nvalue(i))]] + [
                    [qml.Identity(get_nvalue(n)), qml.PauliZ(get_nvalue(n))] for n in neighbours
                ]
                n_coeffs = [[1, sign] for n in neighbours]

                final_terms = [qml.operation.Tensor(*list(m)).prune() for m in itertools.product(*n_terms)]
                final_coeffs = [
                    (0.5**degree) * functools.reduce(lambda x, y: x * y, list(m), 1)
                    for m in itertools.product(*n_coeffs)
                ]

                coeffs.extend(final_coeffs)
                terms.extend(final_terms)
            elif u < v and not graph.has_edge(u,v):
                neighbours_u = sorted(graph.neighbors(u)) if is_rx else list(graph.neighbors(u))
                neighbours_v = sorted(graph.neighbors(v)) if is_rx else list(graph.neighbors(v))
                neighbours   = sorted(list(set(neighbours_u) | set(neighbours_v)))
                degree       = len(neighbours)
                n_terms      = [[qml.PauliX(get_nvalue(u))@qml.PauliX(get_nvalue(v))]] +[[qml.Identity(get_nvalue(n)), qml.PauliZ(get_nvalue(n))] for n in neighbours]
                n_coeffs     = [[1, sign] for n in neighbours]
                final_terms = [qml.operation.Tensor(*list(m)).prune() for m in itertools.product(*n_terms)]
                final_coeffs = [
                    (0.5**degree) * functools.reduce(lambda x, y: x * y, list(m), 1)
                    for m in itertools.product(*n_coeffs)
                ]

                coeffs.extend(final_coeffs)
                terms.extend(final_terms)
            
            
            elif u < v and graph.has_edge(u,v):
                neighbours_u = sorted(graph.neighbors(u)) if is_rx else list(graph.neighbors(u))
                neighbours_v = sorted(graph.neighbors(v)) if is_rx else list(graph.neighbors(v))
                neighbours   = sorted(list(set(neighbours_u) | set(neighbours_v))) 
                neighbours.remove(u)
                neighbours.remove(v)
                degree       = len(neighbours)
                
                
                n_terms      = [[qml.PauliX(get_nvalue(u))@qml.PauliX(get_nvalue(v))]] +[[qml.Identity(get_nvalue(n)), qml.PauliZ(get_nvalue(n))] for n in neighbours]
                n_coeffs     = [[1, sign] for n in neighbours]
                final_terms = [qml.operation.Tensor(*list(m)).prune() for m in itertools.product(*n_terms)]
                final_coeffs = [
                    (0.5**(degree+1)) * functools.reduce(lambda x, y: x * y, list(m), 1)
                    for m in itertools.product(*n_coeffs)
                ]

                coeffs.extend(final_coeffs)
                terms.extend(final_terms)
                
                n_terms      = [[qml.PauliY(get_nvalue(u))@qml.PauliY(get_nvalue(v))]] +[[qml.Identity(get_nvalue(n)), qml.PauliZ(get_nvalue(n))] for n in neighbours]
                n_coeffs     = [[1, sign] for n in neighbours]
                final_terms = [qml.operation.Tensor(*list(m)).prune() for m in itertools.product(*n_terms)]
                final_coeffs = [
                  (0.5**(degree+1)) * functools.reduce(lambda x, y: x * y, list(m), 1)
                    for m in itertools.product(*n_coeffs)
                ]

                coeffs.extend(final_coeffs)
                terms.extend(final_terms)
            
                
    return qml.Hamiltonian(coeffs, terms)