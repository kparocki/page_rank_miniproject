#Code Authors:
#Simon Vanting Daneesøe
#Malthe Have Musaeus
#Krzysztof Parocki

import numpy as np
import networkx as nx
import matplotlib as plt
import random as random
import re

def main():
  #INITIALIZATION --------------------------------------------
  filename = "p2p-Gnutella08-mod.txt"

  def read_file_to_graph(filename):
    '''Open a file and return a directed graph'''
    with open(filename, 'rb') as file:
      G=nx.read_adjlist(file, create_using=nx.DiGraph())

    return G


  # Load in the graph and compute the length (number of nodes)
  graph = read_file_to_graph(filename)

  # Compute backlinks
  branching = []
  for page in graph.nodes():
    branching.append(graph.in_degree(page))

  def compute_dangling_nodes(graph):
    """
    Computes the dangling nodes in the graph.
    """
    dangling_nodes = []
    for page in graph.nodes():
      if graph.out_degree(page) == 0:
        dangling_nodes.append(page)
    return np.asarray(dangling_nodes)

  #get the number of nodes by reading it of the first line of the file
  with open (filename) as f:
      size = f.readline()
  find_list = re.findall('[0-9]+', size)
  number_of_nodes = int(find_list[0])

  #RANDOM SURFER --------------------------------------------

  #iterations coefficient
  #runtime about 13 minutes for number_of_nodes*1000 (about 60 000 000) iterations
  n = 10000

  #start on a random node
  visited_nodes = np.empty(number_of_nodes*n)
  m = 0.15 #damping factor
  arr = np.asarray(graph.nodes)
  current_node = np.random.choice(arr)
  visited_nodes[0] = current_node

  #random walk using successors
  #from experience - *100 is needed for the results to 'converge' somewhat
  #from experience - *10000 is needed for the results to completely stabilize
  for i in range (1,(number_of_nodes)*n):

      if random.random()<(1-m):
          choice = np.asarray(list(graph.successors(current_node)))
          if choice.size == 0:
              current_node = np.random.choice(arr)
          else:
              current_node = np.random.choice(choice)

      else:
          current_node = np.random.choice(arr)

      visited_nodes[i] = current_node

  #count and sort descending depending on counts
  nodes_indexes, counts = np.unique(visited_nodes, return_counts = True)
  c = sorted(zip(counts,nodes_indexes), reverse = True)

  print('\nRANDOM SURFER RESULTS')
  print("The 10 most often visited pages are:")
  for visits, node in c[:10]:
    print(f'-Visits: {visits}, page: {int(node)}')

  #PAGE RANK --------------------------------------------

  # Create X_0
  # Compute initial importance vector (1/n)
  n = 1/number_of_nodes
  x0 = np.full((number_of_nodes,1), n)

  # Compute matrix A
  A = np.zeros((number_of_nodes, number_of_nodes))
  for page in graph.nodes():
      # Get all out links of page
      for out_link in graph.out_edges(page):
          A[int(out_link[1]), int(page)] = 1 / graph.out_degree(page)

  #compute D
  D = np.zeros((number_of_nodes, number_of_nodes))
  dangling_nodes = compute_dangling_nodes(graph)
  # Loop over all columns in D
  for page in graph.nodes():
      if page in list(dangling_nodes):
          D[:, int(page)] = 1/number_of_nodes

  #We don't really need to compute S, as S*x_k always = x0

  # Damping factor = m
  m = 0.15
  m_rev = 1-m

  # Start by defining a variable (current_x_k) that holds the current X_k value
  current_x_k = x0
  AD = np.add(A,D) #we decided to add these first, as we are using vectors anyway 
  #(and thus we don't have the need for two loops to get every current_x_k component)

  #initial M * x0
  M = m_rev*(np.matmul(AD, current_x_k))+m*(x0)

  #interation count to see when it converges
  count = 0

  #we subtract two vectors and check if all entries are 0 with .any()
  while (np.subtract(current_x_k, M).any()):

    #We are never really computing M because we perform calculations
    #on A,D and S instead to optimize our algorithm
    current_x_k = M

    #compute the new M * xk for while condition
    M = m_rev*(np.matmul(AD, current_x_k))+m*(x0)

    count+=1

  #create a numerically ordered list of ints corresponding to nodes' numbers
  #instead of graph.nodes that gives an iterable that is 1)only strings 2)unordered
  ordered_nodes = [*range(0,number_of_nodes)]

  # Compute the 10 highest ranked pages along with their ranks
  sorted_pages = sorted(zip(current_x_k[:,0], ordered_nodes), reverse = True)

  print('\nPAGE RANK RESULTS')	 
  print("The 10 highest ranked pages are:")
  for rank, page in sorted_pages[:10]:
    print(f'-Rank: {rank}, page: {page}')
  print(f"\nConverges after {count} iterations")
  print("\nFinal importance vector:")
  print(current_x_k)
  print('The sum of entries (should be close to 1):')
  print(np.sum(current_x_k))

if __name__ == '__main__':
  main()