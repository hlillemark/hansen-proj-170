import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob
import matplotlib as plt
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
    # init return arrays
    edges_to_remove = []
    nodes_to_remove = []
    # memo[x][0]: shortest path for x
    # memo[x][1]: second shortest path for x
    # memo[x][2]: path for shortest path for x
    # memo[x][3]: path for second shortest path for x
    memo = [[float("inf"), float("inf"), [], []]] * (end + 1)

    # print("out")
    # x = nx.all_shortest_paths(G, 24, 0, weight="weight")
    # print(list(x))
    
    # output of dijkstra will be graph of length of shortest path from one node to the final node and 
    # the path that it took
    shortest_paths = dijkstra(G2)
    offset = 0
    for i in range(c):
        print('DAFUNX')
        node_to_remove = second_best_node(G2, shortest_paths)
        
        if (node_to_remove[1] == 0):
            print('breaking on ')
            print(node_to_remove)
            break
        nodes_to_remove.append(node_to_remove[1] + offset)
        G2.remove_node(node_to_remove[1])
        offset += 1
        end = len(G2.nodes)
        mapping = {}
        for i in range(node_to_remove[1] + 1, end):
            mapping[i] = i - 1
        G2 = nx.relabel_nodes(G2, mapping)
        shortest_paths = dijkstra(G2) # TODO Fix this shit
    
    # print(nodes_to_remove)
        
    # shortest_paths = dijkstra(G2)

    # find shortest path from start to finish, and analyze what the second best path would be for deleting that edge
    # whichever of these second best paths is worst should be returned 
    for i in range(k):
        edge_to_remove = second_best_edge(G2, shortest_paths)
        # print(edge_to_remove)
        # if edge_to_remove[1] == 0:
        #     edges_to_remove.pop(len(edges_to_remove) - 1)
        #     break

        # if edge_to_remove[0] == float('inf'):
            # edges_to_remove.pop(len(edges_to_remove)-1)
            # print('broke')
            # break
        shortest_paths[0][0] = edge_to_remove[0]
        shortest_paths[0][1] = edge_to_remove[1]
        edges_to_remove.append([edge_to_remove[2][0] + offset, edge_to_remove[2][1] + offset])
        # print(edge_to_remove)
        G2.remove_edge(edge_to_remove[2][0], edge_to_remove[2][1])
        # shortest_paths = dijkstra(G2)
        if (not shortest_paths[0][1]):
            # print(edges_to_remove)
            edges_to_remove.pop(len(edges_to_remove) - 1)
            break
    # print(shortest_paths)
    # print(edges_to_remove)
    return nodes_to_remove, edges_to_remove
    

def second_best_node(G, shortest_paths):
    nodes = []
    # collection of [0,x] paths along the way from 0 to t
    new_min = [float('inf'), 0]
    # collection of []
    number_of_vertices = len(G.nodes)
    
    for i in range(1, len(shortest_paths[0][1])- 1):
        if (G.has_node(shortest_paths[0][1][i])):
            nodes.append(shortest_paths[0][1][i])
    pdb.set_trace()
    for node in nodes:
        # pdb.set_trace()
        G2 = G.copy()
        G2.remove_node(node)
        
        end = len(G2.nodes)
        mapping = {}
        for i in range(node + 1, end):
            mapping[i] = i - 1
        G2 = nx.relabel_nodes(G2, mapping)
        new_shortest_paths = dijkstra(G2)
        # pdb.set_trace()
        if (new_shortest_paths[0][0] < new_min[0]):
            new_min = [new_shortest_paths[0][0], node]
    
    return new_min



def second_best_edge(G, shortest_paths):
    source_paths = []
    # collection of [0,x] paths along the way from 0 to t
    new_min = [float('inf'), 0, 0]
    # collection of []
    number_of_vertices = len(G.nodes)

    
    for i in range(len(shortest_paths[0][1]) - 1):
        source_paths.append([shortest_paths[0][1][i], shortest_paths[0][1][i+1]])
    # print("Bruh")
    # print(source_paths)
    path_so_far = [0]
    for i, path in enumerate(source_paths):
        # delete that path, so the previous index where that is coming from should choose the shortest vertex
        # that is not the one that was just deleted
        s = path[0]
        t = path[1]
        cur_min = [float('inf'), 0, 0]
        for neighbor_index in range(number_of_vertices):
            if G.has_edge(s, neighbor_index) and neighbor_index != t:  #path exists and not same one
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
    # pdb.set_trace()

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

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    c, k = solve(G)
    assert is_valid_solution(G, c, k)
    print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
    write_output_file(G, c, k, 'my_out/30.out')


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
# if __name__ == '__main__':
#     inputs = glob.glob('inputs/large/*')
#     for input_path in inputs:
#         output_path = 'outputs/large/' + basename(normpath(input_path))[:-3] + '.out'
#         G = read_input_file(input_path)
#         c, k = solve(G)
#         assert is_valid_solution(G, c, k)
#         distance = calculate_score(G, c, k)
#         write_output_file(G, c, k, output_path)
