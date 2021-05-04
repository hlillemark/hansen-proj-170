import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob
import matplotlib.pyplot as plt
import pdb


def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        c: list of cities to remove
        k: list of edges to remove
    """
    # information about G
    G2 = G.copy()
    nodes = G2.nodes
    edges = G2.edges
    start = 0
    end = len(nodes) - 1
    V = len(nodes)
    # print("G")
    # print(G.adj[0][5])
    # memoized version of the outputs - for each v on the path from s to t, min length from v to t, and path from v to t
    # saved in an array. So for each v, we know about which deletion would be the most impactful. 
    # saved shortest path tree, and then can update that accordingly
    # Init table: length of best and second best path for last:
    # set correct k and c values 
    if (end + 1) <= 30:
        k = 15
        c = 1
    elif (end + 1) <= 50:
        k = 50
        c = 3
    else:
        k = 100
        c = 5
    k_og = k
    # init return arrays
    edges_to_remove = []
    nodes_to_remove = []
    # memo[x][0]: shortest path for x
    # memo[x][1]: second shortest path for x
    # memo[x][2]: path for shortest path for x
    # memo[x][3]: path for second shortest path for x
    memo = [[float("inf"), float("inf"), [], []]] * (end + 1)
    shortest_paths = dijkstra(G2)
    unremovable_edges = []
    mapping = {}
    for i in range(len(G.nodes)):
        mapping[i] = i
    # For amount of possible edges to remove, remove all edges to maximize 
    # Length travelled 
    while k != 0:
        edge_to_remove = remove_best_edge(G2, shortest_paths, unremovable_edges, V)
        if edge_to_remove:
            edges_to_remove.append(edge_to_remove)
            G2.remove_edge(edge_to_remove[0], edge_to_remove[1])
            shortest_paths = dijkstra(G2)
            k -=1
        else:
            break

    # Find the nodes most commonly seen in the edges to remove matrix, and 
    # remove the most common node(s), as long as there is still a viable path
    
    nodes_to_remove_prelim, edges_associated = find_best_nodes(G2, edges_to_remove, c)
    print(edges_associated)
    G3 = G.copy()
    for node in nodes_to_remove_prelim:
        G4 = G3.copy()
        
        # remove = [(n1, n2) for n1,n2,w in G4.edges(node, data=True)]
        # G4.remove_edges_from(remove)
        # G4.remove_node(node)
        # G4.add_node(node)
        G4.remove_node(node)
        G4.add_node(node)
        shortest_paths = dijkstra(G4)
        if (not shortest_paths[0][1]): # can't reach finish any more
            continue
        else:
            G3 = G4
            nodes_to_remove.append(node)
            for mapping_node in mapping.keys():
                if (mapping_node >= node):
                    mapping[mapping_node] += 1
            for edge in edges_associated[node]:
                if (edge in edges_to_remove): # other side could have been removed
                    edges_to_remove.remove(edge)
                    k += 1
            c -=1
    # G5 = G.copy()
    
    # for edge in edges_to_remove:
    #     G5.remove_edge(edge[0], edge[1])
    edges_to_remove = []
    # for node in nodes_to_remove:
    #     G5.remove_node(node)
    # For the amount of nodes removed, remove a corresponding number of extra
    # edges, as long as there is still a viable path, as long as they are not 
    # in the list of nodes that were removed
    shortest_paths = dijkstra(G3)
    unremovable_edges = []
    k = k_og
    while k != 0:
        edge_to_remove = remove_best_edge(G3, shortest_paths, unremovable_edges, V)
        if edge_to_remove:
            # shifted_edge_to_remove = (mapping[edge_to_remove[0]], mapping[edge_to_remove[1]])
            # edges_to_remove.append(shifted_edge_to_remove)
            edges_to_remove.append(edge_to_remove)
            G3.remove_edge(edge_to_remove[0], edge_to_remove[1])
            shortest_paths = dijkstra(G3)
            k -=1
        else:
            break
    # nx.draw(G3)
    # plt.show()
    

    # ##############
    # output of dijkstra will be graph of length of shortest path from one node to the final node and 
    # the path that it took
    # shortest_paths = dijkstra(G2)
    # offset = 0
    # offset_vals = [0] * len(nodes)
    # unremovable_edges = []
    # while c + k > 0:
    #     if c != 0: # try to remove node
    #         # print('trying to delete node...')
    #         node_to_remove = second_best_node(G2, shortest_paths)
    #         # print(node_to_remove)
    #         if (node_to_remove[1] == 0):
    #             # print('no node to delete... breaking on ')
    #             # print(shortest_paths)
    #             # print(c, k)
    #             if k == 0:
    #                 break
    #         else:
    #             # print('deleting node...')
    #             # print(node_to_remove)
    #             nodes_to_remove.append(node_to_remove[1] + offset_vals[node_to_remove[1]])
    #             # print(node_to_remove[1] + offset_vals[node_to_remove[1]])
    #             G2.remove_node(node_to_remove[1])
    #             # for i, x in enumerate(offset_vals):
    #                 # if i >= node_to_remove[1] + offset_vals[i] - 1:
    #                 #     offset_vals[i] = offset_vals[i] + 1
    #             for i in range(node_to_remove[1], len(G2.nodes)):
    #                 offset_vals[i] = offset_vals[i + 1] + 1
                
    #             # offset += 1
    #             end = len(G2.nodes)
    #             mapping = {}
    #             for i in range(node_to_remove[1] + 1, end + 1):
    #                 mapping[i] = i - 1
    #             G2 = nx.relabel_nodes(G2, mapping)
    #             shortest_paths = dijkstra(G2) # TODO Fix this shit
    #             c -= 1
    #             # print(node_to_remove[1] + offset_vals[node_to_remove[1]])

    #     if k != 0: #try to remove edge
    #         # print('trying to delete edge...')
    #         edge_to_remove = second_best_edge(G2, shortest_paths, unremovable_edges)
    #         if (edge_to_remove[2] == 0):
    #             break
    #         # print(edge_to_remove)
    #         # if edge_to_remove[1] == 0:
    #         #     edges_to_remove.pop(len(edges_to_remove) - 1)
    #         #     break

    #         # if edge_to_remove[0] == float('inf'):
    #             # edges_to_remove.pop(len(edges_to_remove)-1)
    #             # print('broke')
    #             # break
    #         # shortest_paths[0][0] = edge_to_remove[0]
    #         # shortest_paths[0][1] = edge_to_remove[1]
    #         # print('removing edge:')
    #         # print(edge_to_remove)
    #         # print(edge_to_remove)
    #         G3 = G2.copy()
    #         G3.remove_edge(edge_to_remove[2][0], 
    #                         edge_to_remove[2][1])
    #         shortest_paths = dijkstra(G3)
    #         if (not shortest_paths[0][1]): # can't reach finish any more
    #             # print(edges_to_remove)
    #             # break
    #             unremovable_edges.append(edge_to_remove[2])
    #             shortest_paths = dijkstra(G2)
    #         else:
    #             # print(offset_vals)
    #             # edges_to_remove.append([edge_to_remove[2][0] + offset_vals[edge_to_remove[2][0] + offset_vals[edge_to_remove[2][0]]], 
    #             #                         edge_to_remove[2][1] + offset_vals[edge_to_remove[2][1] + offset_vals[edge_to_remove[2][1]]]])
    #             edges_to_remove.append([edge_to_remove[2][0] + offset_vals[edge_to_remove[2][0]], 
    #                                     edge_to_remove[2][1] + offset_vals[edge_to_remove[2][1]]])
    #             G2 = G3
                
    #             k -= 1


    # #####################

    # for i in range(c):
    #     print('DAFUNX')
    #     node_to_remove = second_best_node(G2, shortest_paths)
        
    #     if (node_to_remove[1] == 0):
    #         print('breaking on ')
    #         print(node_to_remove)
    #         break
    #     nodes_to_remove.append(node_to_remove[1] + offset)
    #     G2.remove_node(node_to_remove[1])
    #     offset += 1
    #     end = len(G2.nodes)
    #     mapping = {}
    #     for i in range(node_to_remove[1] + 1, end):
    #         mapping[i] = i - 1
    #     G2 = nx.relabel_nodes(G2, mapping)
    #     shortest_paths = dijkstra(G2) # TODO Fix this shit
    
    # print(nodes_to_remove)
        
    # shortest_paths = dijkstra(G2)

    # find shortest path from start to finish, and analyze what the second best path would be for deleting that edge
    # whichever of these second best paths is worst should be returned 
    # for i in range(k):
    #     edge_to_remove = second_best_edge(G2, shortest_paths)
    #     # print(edge_to_remove)
    #     # if edge_to_remove[1] == 0:
    #     #     edges_to_remove.pop(len(edges_to_remove) - 1)
    #     #     break

    #     # if edge_to_remove[0] == float('inf'):
    #         # edges_to_remove.pop(len(edges_to_remove)-1)
    #         # print('broke')
    #         # break
    #     shortest_paths[0][0] = edge_to_remove[0]
    #     shortest_paths[0][1] = edge_to_remove[1]
    #     edges_to_remove.append([edge_to_remove[2][0] + offset, edge_to_remove[2][1] + offset])
    #     # print(edge_to_remove)
    #     G2.remove_edge(edge_to_remove[2][0], edge_to_remove[2][1])
    #     shortest_paths = dijkstra(G2)
    #     if (not shortest_paths[0][1]):
    #         # print(edges_to_remove)
    #         edges_to_remove.pop(len(edges_to_remove) - 1)
    #         break
    # print(shortest_paths)
    # print(edges_to_remove)
    # print(c)
    # print(shortest_paths)
    # print(G.edges)
    # print(nodes_to_remove)
    # print(edges_to_remove)
    # pdb.set_trace()
    return nodes_to_remove, edges_to_remove


# def remove(G)
    
def find_best_nodes(G, edges_to_remove, c):
    node_uses = {}
    node_uses_list = {}
    for i in range(len(G.nodes)):
        # node_uses[i] = 0
        node_uses[i] = []

    for edge in edges_to_remove:
        l = edge[0]
        r = edge[1]
        # node_uses[l] += 1
        # node_uses[r] += 1
        node_uses[l].append(edge) # keep track of what edge deletions would be
        node_uses[r].append(edge) # redundant by deleting that node
    
    node_uses.pop(0)
    node_uses.pop(len(G.nodes) - 1)

    # sorted_node_uses = sorted(dict1, key=dict1.get)
    sorted_node_uses = sorted(node_uses, key=lambda x: len(node_uses.get(x)))
    sorted_node_uses.reverse()
    sorted_node_uses_list = {}
    for node in sorted_node_uses[0:c]:
        sorted_node_uses_list[node] = node_uses[node]
    return sorted_node_uses[0:c], sorted_node_uses_list



def remove_best_node(G, shortest_paths):
    print('removing node')
    # nodes_to_remove = []
    node_to_remove = second_best_node(G, shortest_paths, V)
    
    if (node_to_remove[1] == 0):
        print('breaking on ')
        print(node_to_remove)
        return None


# def remove_best_edge(G, shortest_paths, unremovable_edges):
#     edge_to_remove = second_best_edge(G, shortest_paths, unremovable_edges)
#     if (edge_to_remove[2] == 0): # can't find edge to remove
#         return None
#     G2 = G.copy()
#     # Try removing edge and see if viable path:
#     G2.remove_edge(edge_to_remove[2][0], 
#                     edge_to_remove[2][1])
#     shortest_paths = dijkstra(G2)
#     if (not shortest_paths[0][1]): # can't reach finish any more
#         # print(edges_to_remove)
#         # break
#         unremovable_edges.append(edge_to_remove[2])
#         shortest_paths = dijkstra(G)
#         result = remove_best_edge(G, shortest_paths, unremovable_edges)
#         if not result:
#             return None
#         else:
#             return result
#     else:
#         return G2
#         # print(offset_vals)
#         # edges_to_remove.append([edge_to_remove[2][0] + offset_vals[edge_to_remove[2][0] + offset_vals[edge_to_remove[2][0]]], 
#         #                         edge_to_remove[2][1] + offset_vals[edge_to_remove[2][1] + offset_vals[edge_to_remove[2][1]]]])
#         # edges_to_remove.append([edge_to_remove[2][0] + offset_vals[edge_to_remove[2][0]], 
#         #                         edge_to_remove[2][1] + offset_vals[edge_to_remove[2][1]]])
        
#     # print('removing edge')


def remove_best_edge(O_G, shortest_paths, unremovable_edges, V):
    cur_unremovable = unremovable_edges.copy()
    C_G = O_G.copy()
    while True:
        edge_to_remove = second_best_edge(C_G, shortest_paths, cur_unremovable)
        if (edge_to_remove[2] == 0):
            return None
        # Try removing edge and see if viable path:
        # print('trying ', edge_to_remove[2])
        C_G.remove_edge(edge_to_remove[2][0], 
                        edge_to_remove[2][1])
        shortest_paths = dijkstra(C_G)
        if (not shortest_paths[0][1]): # can't reach finish any more
            # print(edges_to_remove)
            # break
            cur_unremovable.append(edge_to_remove[2])
            C_G = O_G.copy()
            shortest_paths = dijkstra(C_G)
        else:
            return edge_to_remove[2]
            # print(offset_vals)
            # edges_to_remove.append([edge_to_remove[2][0] + offset_vals[edge_to_remove[2][0] + offset_vals[edge_to_remove[2][0]]], 
            #                         edge_to_remove[2][1] + offset_vals[edge_to_remove[2][1] + offset_vals[edge_to_remove[2][1]]]])
            # edges_to_remove.append([edge_to_remove[2][0] + offset_vals[edge_to_remove[2][0]], 
            #                         edge_to_remove[2][1] + offset_vals[edge_to_remove[2][1]]])
        



def nodes_along_shortest_paths(G, shortest_paths):
    nodes = []
    for i in range(1, len(shortest_paths[0][1])- 1):
        if (G.has_node(shortest_paths[0][1][i])):
            nodes.append(shortest_paths[0][1][i])
    return nodes

def edges_along_shortest_path(G, shortest_paths):
    edges = []
    for i in range(len(shortest_paths[0][1]) - 1):
        edges.append((shortest_paths[0][1][i], shortest_paths[0][1][i+1]))
    return edges


def second_best_node(G, shortest_paths, V):
    nodes = nodes_along_shortest_paths(G, shortest_paths)
    # collection of [0,x] paths along the way from 0 to t
    new_min = [sys.maxsize, 0]
    # collection of []
    
    for node in nodes:
        
        G2 = G.copy()
        G2.remove_node(node)
        
        end = len(G2.nodes)
        mapping = {}
        for i in range(node + 1, end + 1):
            mapping[i] = i - 1
        G2 = nx.relabel_nodes(G2, mapping)
        new_shortest_paths = dijkstra(G2)
        if (new_shortest_paths[0][0] < new_min[0]):
            new_min = [new_shortest_paths[0][0], node]
    
    return new_min



def second_best_edge(G, shortest_paths, unremovable_edges):
    source_paths_og = edges_along_shortest_path(G, shortest_paths)
    source_paths = []
    for path in source_paths_og:
        if path not in unremovable_edges:
            source_paths.append(path)
                
    # collection of [0,x] paths along the way from 0 to t
    new_min = [sys.maxsize, 0, 0]
    # collection of []
    number_of_vertices = len(G.nodes)

    
    # print("Bruh")
    # print(source_paths)
    path_so_far = [0]
    for i, path in enumerate(source_paths):
        # delete that path, so the previous index where that is coming from should choose the shortest vertex
        # that is not the one that was just deleted
        s = path[0]
        t = path[1]
        cur_min = [sys.maxsize, 0, 0]
        for neighbor_index in range(number_of_vertices):
            if G.has_edge(s, neighbor_index) and neighbor_index != t:  #path exists and not same one
                G2 = G.copy()
                new_shortest_paths = dijkstra(G2)
                # if (new_shortest_paths[0][0] < cur_min[0]):
                #     cur_min = [new_shortest_paths[0][0], path]
                # print('path')
                # print(shortest_paths[neighbor_index])
                if ((path not in shortest_paths[neighbor_index][1]) and 
                   ([t + s] not in shortest_paths[neighbor_index][1])):     # replacement does not use deleted path
                    if (G[s][neighbor_index]['weight'] + shortest_paths[neighbor_index][0] < cur_min[0]):
                        cur_min = ([G[s][neighbor_index]['weight'] + shortest_paths[neighbor_index][0], 
                                        path_so_far + shortest_paths[neighbor_index][1], path])
        if cur_min[0] < new_min[0]:
            new_min = cur_min
    return new_min # currently does not get rid of deleted paths?


# def dijkstra(G, V):
#     # information about G
#     nodes = G.nodes
#     end = V - 1
#     number_of_vertices = V
#     # for v in range(start + 1):
#     def to_be_visited(visited_and_distance):
#         # global visited_and_distance
#         v = -10
#           # Choosing the vertex with the minimum distance
#         for index in range(number_of_vertices):
#             if visited_and_distance[index][0] == 0 \
#             and (v < 0 or visited_and_distance[index][1] <= \
#             visited_and_distance[v][1]):
#                 v = index
#         return v

#     # The first element of the lists inside visited_and_distance 
#     # denotes if the vertex has been visited.
#     # The second element of the lists inside the visited_and_distance 
#     # denotes the distance from the source.
#     visited_and_distance = [[0, 0, [end]]]
#     for i in range(number_of_vertices-1):
#         visited_and_distance.insert(0, [0, sys.maxsize, []])

#     for vertex in range(number_of_vertices):
#     # Finding the next vertex to be visited.
#         to_visit = to_be_visited(visited_and_distance)
#         # print("to_visit")
#         # print(to_visit)
#         # for 
#         for neighbor_index in range(number_of_vertices):
#             # Calculating the new distance for all unvisited neighbours
#             # of the chosen vertex.
#             # if vertices[to_visit][neighbor_index] == 1 and \
#             # visited_and_distance[neighbor_index][0] == 0:
#             if G.has_edge(to_visit, neighbor_index) and \
#             visited_and_distance[neighbor_index][0] == 0:
#                 new_distance = visited_and_distance[to_visit][1] \
#                 + G[to_visit][neighbor_index]['weight']
#                 new_path = [neighbor_index] + visited_and_distance[to_visit][2]
#             else: 
#                 continue
#                 # + edges[to_visit][neighbor_index]
#             # Updating the distance of the neighbor if its current distance
#             # is greater than the distance that has just been calculated
#             if visited_and_distance[neighbor_index][1] > new_distance:
#                 visited_and_distance[neighbor_index][1] = new_distance
#                 visited_and_distance[neighbor_index][2] = new_path
#         # Visiting the vertex found earlier
#         visited_and_distance[to_visit][0] = 1

#     i = 0 
#     return [[distance[1], distance[2]] for distance in visited_and_distance]
#     # Printing out the shortest distance from the source to each vertex       
#     for distance in visited_and_distance:
#         print("The shortest distance of ",str(i),\
#         " from the source vertex 24 is:",distance[1], " and the path is", distance[2])
        
#         i = i + 1


def dijkstra(G):
    # information about G
    nodes = G.nodes
    end = len(nodes) - 1
    number_of_vertices = len(nodes)
    # for v in range(start + 1):
    def to_be_visited(visited_and_distance):
        # global visited_and_distance
        v = -10
          # Choosing the vertex with the minimum distance
        for index in range(number_of_vertices):
            if visited_and_distance[index][0] == 0 \
            and (v < 0 or visited_and_distance[index][1] <= \
            visited_and_distance[v][1]):
                v = index
        return v

    # The first element of the lists inside visited_and_distance 
    # denotes if the vertex has been visited.
    # The second element of the lists inside the visited_and_distance 
    # denotes the distance from the source.
    visited_and_distance = [[0, 0, [end]]]
    for i in range(number_of_vertices-1):
        visited_and_distance.insert(0, [0, sys.maxsize, []])

    for vertex in range(number_of_vertices):
    # Finding the next vertex to be visited.
        to_visit = to_be_visited(visited_and_distance)
        # print("to_visit")
        # print(to_visit)
        # for 
        for neighbor_index in range(number_of_vertices):
            # Calculating the new distance for all unvisited neighbours
            # of the chosen vertex.
            # if vertices[to_visit][neighbor_index] == 1 and \
            # visited_and_distance[neighbor_index][0] == 0:
            if G.has_edge(to_visit, neighbor_index) and \
            visited_and_distance[neighbor_index][0] == 0:
                new_distance = visited_and_distance[to_visit][1] \
                + G[to_visit][neighbor_index]['weight']
                new_path = [neighbor_index] + visited_and_distance[to_visit][2]
            else: 
                continue
                # + edges[to_visit][neighbor_index]
            # Updating the distance of the neighbor if its current distance
            # is greater than the distance that has just been calculated
            if visited_and_distance[neighbor_index][1] > new_distance:
                visited_and_distance[neighbor_index][1] = new_distance
                visited_and_distance[neighbor_index][2] = new_path
        # Visiting the vertex found earlier
        visited_and_distance[to_visit][0] = 1

    i = 0 
    return [[distance[1], distance[2]] for distance in visited_and_distance]
    # Printing out the shortest distance from the source to each vertex       
    for distance in visited_and_distance:
        print("The shortest distance of ",str(i),\
        " from the source vertex 24 is:",distance[1], " and the path is", distance[2])
        
        i = i + 1



# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G = read_input_file(path)
#     c, k = solve(G)
#     assert is_valid_solution(G, c, k)
#     print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
#     write_output_file(G, c, k, 'my_out/30.out')


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
# if __name__ == '__main__':
#     inputs = glob.glob('my_in/*')
#     for input_path in inputs:
#         output_path = 'my_out/' + basename(normpath(input_path))[:-3] + '.out'
#         G = read_input_file(input_path)
#         c, k = solve(G)
#         assert is_valid_solution(G, c, k)
#         distance = calculate_score(G, c, k)
#         write_output_file(G, c, k, output_path)

# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
if __name__ == '__main__':
    inputs = glob.glob('inputs/large/*')
    i = 1
    for input_path in inputs:
        print("file ", i, " / 300")
        i += 1
        output_path = 'outputs/large/' + basename(normpath(input_path))[:-3] + '.out'
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        distance = calculate_score(G, c, k)
        write_output_file(G, c, k, output_path)
