import networkx as nx
import matplotlib.pyplot as plt
import heapq

# Завдання 1: Створення графа та аналіз основних характеристик
web_graph = {
    "google.com": {"facebook.com", "youtube.com"},
    "facebook.com": {"google.com", "youtube.com", "amazon.com"},
    "youtube.com": {"google.com", "facebook.com", "amazon.com", "twitter.com"},
    "amazon.com": {"facebook.com", "youtube.com"},
    "twitter.com": {"youtube.com"}
}

G = nx.Graph()
G.add_nodes_from(web_graph.keys())
for node, neighbors in web_graph.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

nx.draw(G, with_labels=True, node_color='skyblue', node_size=800, font_size=10, font_weight='bold')
plt.title("Моделювання графа")
plt.show()

print("Кількість вузлів:", G.number_of_nodes())
print("Кількість ребер:", G.number_of_edges())
print("Ступінь центральності:", nx.degree_centrality(G))

# Завдання 2: DFS та BFS для знаходження шляхів
def dfs_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            new_paths = dfs_paths(graph, node, end, path)
            for new_path in new_paths:
                paths.append(new_path)
    return paths

def bfs_paths(graph, start, end):
    queue = [(start, [start])]
    while queue:
        (node, path) = queue.pop(0)
        for next_node in graph[node] - set(path):
            if next_node == end:
                yield path + [next_node]
            else:
                queue.append((next_node, path + [next_node]))

dfs_paths = dfs_paths(web_graph, "google.com", "twitter.com")
bfs_paths = list(bfs_paths(web_graph, "google.com", "twitter.com"))

print("DFS шляхи:", dfs_paths)
print("BFS шляхи:", bfs_paths)

# Завдання 3: Алгоритм Дейкстри для знаходження найкоротшого шляху
def dijkstra(graph, start):
    shortest_paths = {node: float('inf') for node in graph}
    shortest_paths[start] = 0
    queue = [(0, start)]
    while queue:
        (current_distance, current_node) = heapq.heappop(queue)
        if current_distance > shortest_paths[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < shortest_paths[neighbor]:
                shortest_paths[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    return shortest_paths

web_graph_with_weights = {
    "google.com": {"facebook.com": 1, "youtube.com": 2},
    "facebook.com": {"google.com": 1, "youtube.com": 2, "amazon.com": 1},
    "youtube.com": {"google.com": 2, "facebook.com": 2, "amazon.com": 3, "twitter.com": 1},
    "amazon.com": {"facebook.com": 1, "youtube.com": 3},
    "twitter.com": {"youtube.com": 1}
}

shortest_paths_from_google = dijkstra(web_graph_with_weights, "google.com")
print("Найкоротші шляхи від 'google.com':", shortest_paths_from_google)