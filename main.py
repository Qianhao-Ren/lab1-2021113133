import re
import random
import networkx as nx
import matplotlib.pyplot as plt
import heapq


def preprocess_text(text):
    # 移除非字母字符，将标点视为空格
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 多个空格压缩为一个空格
    text = re.sub(r'\s+', ' ', text)
    # 将文本转换为小写
    return text.lower().strip()


def build_graph(words):
    graph = {}
    for i in range(len(words) - 1):
        node = words[i]
        next_node = words[i + 1]
        if node not in graph:
            graph[node] = {}
        if next_node in graph[node]:
            graph[node][next_node] += 1
        else:
            graph[node][next_node] = 1
    return graph


def find_bridge_words(graph, word1, word2):
    bridge_words = []
    if word1 in graph and word2 in graph:
        for candidate in graph.get(word1, {}):
            if word2 in graph.get(candidate, {}):
                bridge_words.append(candidate)
    return bridge_words


def insert_bridge_words(graph, text):
    words = text.split()
    new_text = words[0]
    for i in range(1, len(words)):
        bridge_words = find_bridge_words(graph, words[i - 1], words[i])
        if bridge_words:
            new_word = random.choice(bridge_words)
            new_text += f" {new_word}"
        new_text += f" {words[i]}"
    return new_text
def dijkstra(G, source, target, weight='weight'):
    # 初始化优先级队列和距离字典
    queue = [(0, source)]
    distances = {node: float('inf') for node in G}
    distances[source] = 0
    previous_nodes = {node: None for node in G}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        # 如果我们到达目标节点，构建并返回路径
        if current_node == target:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = previous_nodes[current_node]
            path.reverse()
            return path

        # 如果从优先级队列中弹出的节点的距离大于已知的最短距离，则跳过它
        if current_distance > distances[current_node]:
            continue

        # 检查邻居并更新距离
        for neighbor, data in G[current_node].items():
            edge_weight = data.get(weight, 1)  # 获取权重，默认为1
            distance = current_distance + edge_weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    # 如果找不到路径，则引发异常
    raise nx.NetworkXNoPath(f"Node {source} not reachable from {target}")

def find_shortest_path(G, source, target):
    try:
        #path = nx.shortest_path(G, source=source, target=target, weight='weight')
        path = dijkstra(G, source, target, weight='weight')
        path_length = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        print(f"The shortest path from {source} to {target} is: {' -> '.join(path)} with total weight {path_length}")
        return path
    except nx.NetworkXNoPath:
        print(f"No path could be found between {source} and {target}.")
        return None
def convert_to_networkx(graph):
    G = nx.DiGraph()
    for node, edges in graph.items():
        for adjacent, weight in edges.items():
            G.add_edge(node, adjacent, weight=weight)
    return G


def plot_graph(G, path=None):
    pos = nx.spring_layout(G)
    edge_labels = dict([((u, v,), d['weight']) for u, v, d in G.edges(data=True)])
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={edge: G[edge[0]][edge[1]]['weight'] for edge in path_edges})
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

def random_walk(G):
    start_node = random.choice(list(G.nodes))
    current_node = start_node
    visited_edges = set()
    walk = [current_node]
    try:
        while True:
            neighbors = list(G[current_node])
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            if (current_node, next_node) in visited_edges:
                break
            visited_edges.add((current_node, next_node))
            walk.append(next_node)
            current_node = next_node
    except KeyboardInterrupt:
        pass
    walk_output = ' '.join(walk)
    with open('random_walk_output.txt', 'w') as file:
        file.write(walk_output)
    print("随机游走已保存到文件 'random_walk_output.txt'")
    return walk_output
def main():
    filename = input("请输入文本文件的路径和文件名：")
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
            processed_text = preprocess_text(text)
            words = processed_text.split()
            graph = build_graph(words)
            G = convert_to_networkx(graph)
            plot_graph(G)
            print("图构建完成，并已展示。")
            # 处理用户的桥接词查询
            while True:
                word1 = input("请输入第一个单词 (输入 'exit' 退出): ").strip().lower()
                if word1 == 'exit':
                    break
                word2 = input("请输入第二个单词: ").strip().lower()
                print(find_bridge_words(graph, word1, word2))
            # 生成新文本
            new_input_text = input("请输入一行新文本：")
            new_output_text = insert_bridge_words(graph, new_input_text)
            print("生成的新文本是：")
            print(new_output_text)
            source = input("请输入起始单词：").strip().lower()
            target = input("请输入目标单词：").strip().lower()
            path = find_shortest_path(G, source, target)
            plot_graph(G, path)
            print("开始随机游走，请按 Ctrl+C 停止...")
            walk_output = random_walk(G)
            print("随机游走的结果是：")
            print(walk_output)
    except Exception as e:
        print(f"读取文件时发生错误：{e}")


if __name__ == '__main__':
    main()
