import heapq
import random
from typing import Any
from collections import defaultdict
from collections.abc import Callable

import networkx as nx

from qmf import durr_hoyer_qmf, makhanov_qmf
from quantum_runner import AerRunner, AzureRunner


def dijkstra(graph: nx.Graph, start: int) -> tuple[dict[int, float], dict[int, int | None]]:
    '''
    A classical implementation of Dijkstra's algorithm.

    Parameters
    ----------
    graph : nx.Graph
        An undirected graph with non-negative edge weights.
    start : int
        The start node.

    Returns
    -------
    distances, shortest_path_tree : tuple[dict[int, float], dict[int, int | None]]
        A dictionary of distances from the start node to each other node.
        A dictionary representing a tree used to reconstruct the shortest path from the start to any other node.
    '''

    # Initialize distances and the priority queue
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    priority_queue = [(0, start)]
    shortest_path_tree = {start: None}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Nodes can get added to the priority queue multiple times. We only
        # process a vertex the first time we remove it from the priority queue.
        if current_distance > distances[current_node]:
            continue

        # Visit each adjacent node
        for neighbour in graph.neighbors(current_node):
            edge_weight = graph[current_node][neighbour]['weight']
            distance = current_distance + edge_weight

            # Only consider this new path if it's better
            if distance < distances[neighbour]:
                distances[neighbour] = distance
                shortest_path_tree[neighbour] = current_node
                heapq.heappush(priority_queue, (distance, neighbour))

    return dict(distances), shortest_path_tree


def dijkstra_qmf(
    graph: nx.Graph, start: int, qmf_algo: Callable, *qmf_args: Any, **qmf_kwargs: Any
) -> tuple[dict[int, float], dict[int, int | None]]:
    '''
    The extended Dijkstra's algorithm.

    Parameters
    ----------
    graph : nx.Graph
        An undirected graph with non-negative edge weights.
    start : int
        The start node.
    qmf_algo : Callable
        An implementation of QMF.
    *qmf_args : Any
    **qmf_kwargs : Any
        Arguments and keyword arguments passed to `qmf_algo`.

    Returns
    -------
    distances, shortest_path_tree : tuple[dict[int, float], dict[int, int | None]]
        A dictionary of distances from the start node to each other node.
        A dictionary representing a tree used to reconstruct the shortest path from the start to any other node.
    '''

    # Initialize distances and the priority queue
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    priority_queue = [(0, start)]
    shortest_path_tree = {start: None}

    while priority_queue:
        min_index, current_distance = qmf_algo([d for d, _ in priority_queue], *qmf_args, **qmf_kwargs)
        current_node = priority_queue.pop(min_index)[1]  # Removes element at `min_index` from the queue

        # Nodes can get added to the priority queue multiple times. We only
        # process a vertex the first time we remove it from the priority queue.
        if current_distance > distances[current_node]:
            continue

        # Visit each adjacent node
        for neighbour in graph.neighbors(current_node):
            edge_weight = graph[current_node][neighbour]['weight']
            distance = current_distance + edge_weight

            # Only consider this new path if it's better
            if distance < distances[neighbour]:
                distances[neighbour] = distance
                shortest_path_tree[neighbour] = current_node
                # heapq.heappush(priority_queue, (distance, neighbour))
                priority_queue.append((distance, neighbour))

    distances.default_factory = None
    return distances, shortest_path_tree


def generate_graph(num_nodes: int, sparsity: float = 0.5) -> nx.Graph:
    '''
    Generate an undirected graph with random edge weights.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    sparsity : float
        Controls the sparsity of edges in the graph.

    Returns
    -------
    nx.Graph
    '''

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < 1.0 - sparsity:
                graph.add_edge(i, j, weight=random.randint(1, 100))
    return graph


def visualize_graph(graph: nx.Graph) -> None:
    '''Plot a graph with matplotlib.'''

    import matplotlib.pyplot as plt

    pos = nx.spring_layout(graph)
    edge_labels = {(u, v): d["weight"] for u, v, d in graph.edges(data=True)}
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color="skyblue")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()


def reconstruct_shortest_path(shortest_path_tree, end):
    '''
    Reconstruct the shortest path from the start node to the specified end node using
    the shortest path tree returned by Dijkstra's algorithm.
    '''

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = shortest_path_tree[current]
    return path[::-1]


if __name__ == '__main__':
    random.seed(1223334444)
    graph = generate_graph(15, sparsity=0.5)
    visualize_graph(graph)

    distances, shortest_path_tree = dijkstra(graph, 0)
    print(reconstruct_shortest_path(shortest_path_tree, 9))

    # DH algo succeeds with just a single shot (on ideal simulator)
    quantum_runner = AzureRunner('ionq.qpu', shots=1, dryrun=True)
    distances, shortest_path_tree = dijkstra_qmf(graph, 0, durr_hoyer_qmf, quantum_runner=quantum_runner)
    print(reconstruct_shortest_path(shortest_path_tree, 9))
    print(quantum_runner.n_calls)
    print(quantum_runner.total_cost, quantum_runner.currency)

    quantum_runner = AzureRunner('ionq.qpu', shots=8, dryrun=True)
    distances, shortest_path_tree = dijkstra_qmf(graph, 0, makhanov_qmf, quantum_runner=quantum_runner, repeats=50)
    print(reconstruct_shortest_path(shortest_path_tree, 9))
    print(quantum_runner.n_calls)
    print(quantum_runner.total_cost, quantum_runner.currency)
