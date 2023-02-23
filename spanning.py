"""spanning.py

In this question your program will read a simple, weighted, undirected, and connected graph,
G(V, E, W), and compute:
  1. a smallest-weight spanning tree of G, and
  2. a second-smallest-weight spanning tree of G, whose weight is greater than or equal to the smallest.

To address this task, you must implement Kruskal’s algorithm by employing union-by-rank
data structure with path compression.

name: Kevin Yu
student_id: 306128531

"""

import sys
import math

# Types
vertex = int
edge_weight = int
edge = [vertex, vertex, edge_weight]

number_edges = int
number_vertices = int

data = ((number_vertices, number_edges), [edge])

total_weight = int
spanning_tree_results = [total_weight, [edge]]


class FileHelper:
  def process_input(filename: str) -> data:
    """processes the file containing metadata and edges

    Args:
        filename (str): filename with the data

    Returns:
        _type_: ((number_vertices, number_edges), [edge])
    """

    with open(filename, "r") as f:
      [number_vertices, number_edges] = [
          int(value) for value in f.readline().rstrip("\n").split(" ")]
      raw_edges = f.readlines()
      edges = [[int(edge_value) for edge_value in edge.rstrip(
          "\n").split(" ")] for edge in raw_edges]

    return ((number_vertices, number_edges), edges)

  def write_results_to_file(
          smallest_spanning_tree_results: spanning_tree_results,
          second_smallest_spanning_tree_results: spanning_tree_results):

    output_filename = "output_spanning.txt"
    copy_smallest = "#List of edges in the smallest spanning tree:\n"
    copy_second_smallest = "#List of edges in the second smallest spanning tree:\n"
    with open(output_filename, "w") as f:
      # Write results for smallest spanning tree
      f.write(
          f'Smallest Spanning Tree Weight = {smallest_spanning_tree_results[0]}\n')
      f.write('#List of edges in the smallest spanning tree:\n')
      f.write(f'')
      for result in smallest_spanning_tree_results[1]:
        f.write(f'{result[0]} {result[1]} {result[2]}\n')

      # Write results for second smallest spanning tree
      f.write(
          f'Second-smallest Spanning Tree Weight = {second_smallest_spanning_tree_results[0]}\n')
      f.write('#List of edges in the second smallest spanning tree:\n')
      f.write(f'')
      for result in second_smallest_spanning_tree_results[1]:
        f.write(f'{result[0]} {result[1]} {result[2]}\n')


class DisjointSet:
  """DisjointSet
  This class represents the DisjointSet data structure
  The methods in this class are adapted from the week 4 lecture slides
  """

  def __init__(self, size: int = 0):
    self.size = size
    # Since the vertex inputs to the algorithm are 1-indexed, our parent array is of size (number of vertices + 1)
    # The 0th index of the self.parent array just never gets changed/used
    self.parent = [-1 for i in range(self.size + 1)]

  def union(self, v_1: int, v_2: int):
    """union

    Performs union on v_1 and v_2

    Args:
        v_1 (int): first vertex
        v_2 (int): second vertex

    Returns:
        _type_: True if the two sets of vertices were "union'd", and False otherwise
        Two sets are only union'd if they don't share the same root (i.e they are disjoint)
    """

    root_v_1 = self.find(v_1)
    root_v_2 = self.find(v_2)

    if (root_v_1 == root_v_2):
      # Return false if the two vertices are in the same set (i.e if they are connected already)
      return False

    height_v_1 = -self.parent[v_1]
    height_v_2 = -self.parent[v_2]

    if height_v_1 > height_v_2:
      self.parent[root_v_2] = root_v_1

    elif height_v_2 > height_v_1:
      self.parent[root_v_1] = root_v_2

    else:
      self.parent[root_v_1] = root_v_2
      self.parent[root_v_2] = -(height_v_2 + 1)

    return True

  def find(self, v: int):
    if (self.parent[v] < 0):
      return v
    else:
      # Changing the parent pointer of all nodes along the path to the root
      self.parent[v] = self.find(v=self.parent[v])
      return self.parent[v]

  def __str__(self):
    return str(self.parent)


class Solution:
  def __init__(self, data: data):
    self.data = data

    # Sort the edges based on their weights
    self.sorted_edges = sorted(self.data[1], key=lambda edge: (edge[2], edge[0], edge[1]))

  def find_min_spanning_tree(
          self, excluded_edge_index: int = None) -> [total_weight, [edge], [int]]:
    """find_min_spanning_tree

    Args:
        excluded_edge_index (int, optional): edges to exclude, used to find second smallest weight MST. Defaults to None.

    Returns:
        [total_weight, [edge], [int]]: total weight and edges in the MST
    """

    ((number_vertices, number_edges), edges) = self.data

    # Initialise the DisjointSet
    disjoint_set = DisjointSet(number_vertices)
    # Stores the edges of our MST
    results = []
    # Stores the indexes of our MST within the sorted list of edges.
    # Used to make find_second_min_spanning_tree a bit more efficient
    results_indexes = []
    total_weight = 0

    for index in range(len(self.sorted_edges)):
      # excluded_edge_index is used solely for the find_second_min_spanning_tree function
      if index != excluded_edge_index:
        edge = self.sorted_edges[index]
        # If we successfully union v_1 and v_2, that means that we have joined them via edge
        # Therefore we add edge to results
        if disjoint_set.union(v_1=edge[0], v_2=edge[1]):
          results.append(edge)
          total_weight += edge[2]
          results_indexes.append(index)

    self.disjoint_set = disjoint_set
    self.accepted_edges = results

    return [total_weight, results, results_indexes]

  def find_second_min_spanning_tree(
          self, results_indexes: [int]) -> [total_weight, [edge]]:
    """find_second_min_spanning_tree

    Args:
        results_indexes ([int]): indexes in self.sorted_edges of edges that are in the MST

    Returns:
        [total_weight, [edge]]: total weight and edges in the MST with second smallest weight
    """

    current_min_weight = math.inf
    current_results = []

    for excluded_edge_index in results_indexes:
      # For each edge in the results, we want to try make an MST without that edge
      # Since the final returned MST can't have the same edges as the smallest MST, it will be unique
      [total_weight, results, _results_indexes] = self.find_min_spanning_tree(
          excluded_edge_index=excluded_edge_index)

      if total_weight < current_min_weight:
        # Update the resultant weights and min weight if it the MST is better than any other we have come across
        current_results = results
        current_min_weight = total_weight

    return [current_min_weight, current_results]


def main(filename: str):
  data = FileHelper.process_input(filename)
  solution = Solution(data)

  [total_weight, results, results_indexes] = solution.find_min_spanning_tree()

  [second_smallest_total_weight, second_smallest_results] = solution.find_second_min_spanning_tree(
      results_indexes=results_indexes)

  # We could remove these sorts as we are already presorting in the order (edge[2], edge[0], edge[1]) before running the algo
  # But I just put it here for peace of mind and makes it more clear that the outputs are definitely sorted
  # Isn't too much more inefficient

  results = sorted(results, key=lambda edge: (edge[0], edge[1]))
  second_smallest_results = sorted(second_smallest_results, key=lambda edge: (edge[0], edge[1]))

  FileHelper.write_results_to_file(
      smallest_spanning_tree_results=[
          total_weight, results], second_smallest_spanning_tree_results=[
          second_smallest_total_weight, second_smallest_results])


if __name__ == "__main__":
  if len(sys.argv[1:]) > 0:
    args = [arg for arg in sys.argv[1:]]

    main(args[0])
