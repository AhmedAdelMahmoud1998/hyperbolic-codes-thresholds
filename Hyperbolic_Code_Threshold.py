import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from multiprocessing import Pool
import multiprocessing
from collections import Counter
import bisect
import time
from networkx.utils import pairwise
from scipy.sparse import coo_matrix, csr_matrix, kron
from sympy.core.containers import OrderedSet

start = time.time()

# This function calculates the hyperbolic distance between two points in the complex plane.
def hyperbolic_distance(z1: complex, z2: complex):
    d = np.arccosh(1 + (2 * (abs(z1 - z2)) ** 2 / ((1 - abs(z1) ** 2) * (1 - abs(z2) ** 2))))
    return d

# This function calculates the Euclidean distance between two points in the complex plane.
def euclidean_distance(z1: complex, z2: complex):
    d = abs(z1 - z2)
    return d

# This function takes as input two integers that define a hyperbolic lattice {p,q} and returns a list of positions for the vertices in the unit cell of that lattice.
def unit_cell_positions(p: int, q: int):
    a = np.pi / p
    b = np.pi / q
    r0 = np.sqrt(np.cos(a + b) / np.cos(a - b))
    vertices_positions = []
    if p == 8 and q == 3:
        for k in range(p):
            vertices_positions.append(r0 * np.exp(1j * np.pi * (2 * k - 1) / p))
        d0 = 0.66
        for k in range(p):
            vertices_positions.append(d0 * np.exp(1j * np.pi * (2 * k - 1) / p))
    else:
        for k in range(p):
            vertices_positions.append(r0 * np.exp(1j * np.pi * (2 * k) / p))
    return vertices_positions

def rotation_matrix(phi: complex):
    return np.array([[np.exp(1j * phi / 2), 0], [0, np.exp(-1j * phi / 2)]])

def fuchsian_generators(p_B: int, q_B: int):
    alpha = 2* np.pi/ p_B
    beta = 2* np.pi/ q_B
    sigma = np.sqrt((np.cos(alpha) + np.cos(beta)) / (1 + np.cos(beta)) )
    gamma1 = 1/(np.sqrt(1-sigma**2)) * np.array([[1, sigma], [sigma,1]])
    FG_generators = []
    for mu in range(0,int(p_B/2)):
        gamma_j = rotation_matrix(mu*alpha) @ gamma1 @ rotation_matrix(-mu*alpha)
        FG_generators.append(gamma_j)
        FG_generators.append(np.linalg.inv(gamma_j))
    return FG_generators

def create_new_vertex(vertex_position: complex, translation_matrix: np.array):
    v = translation_matrix @ np.array([vertex_position, 1])
    new_vertex_position = v[0] / v[1]
    return new_vertex_position

def generate_vertices(p: int,q: int, p_B: int, q_B: int, N: int):
    # Create the unit cell of the {p,q} lattice
    unit_cell = unit_cell_positions(p, q)
    # Calculate the distance between nearest neighbour in this lattice.
    d0 = hyperbolic_distance(unit_cell[0], unit_cell[1])
    # Generate the Fuchsian generators of the Bravais lattice.
    group_generators = fuchsian_generators(p_B, q_B)
    # This list is used to store vertices that are not in the unit cell.
    outer_rings = []
    # This list is used to store redundant vertices. This happens when the vertices in the unit cell lie on the boundaries of the Bravais lattice.
    redundant_indices = []
    
    i = len(unit_cell)-1
    # If N = 1, we return the unit cell.
    if N == 1:
        return unit_cell, redundant_indices
    
    # This is the case when the vertices in the unit cell lie on the boundaries of the Bravais lattice.
    if (p == 8 and q ==4) or (p == 12 and q ==4):
        # We start by producing p_B new faces in the Bravais lattice by applying all the Fuchsian group generators to the unit cell.
        for generator in group_generators:
            for vertex in unit_cell:
                new_vertex = create_new_vertex(vertex, generator)
                i += 1
                # Check if new_vertex is not in any of the vertices in D
                if all(hyperbolic_distance(new_vertex, vertex) > (d0 - 0.15) for vertex in unit_cell + outer_rings):
                    outer_rings.append(new_vertex)
                else:
                    outer_rings.append(new_vertex)
                    redundant_indices.append(i)
            
    # This is the case when all vertices in the unit cell lie inside the Bravais lattice. There is no redundant vertices in this case.
    else:
        # We start by producing p_B new faces in the Bravais lattice by applying all the Fuchsian group generators to the unit cell.
         for generator in group_generators:
            for vertex in unit_cell:
                new_vertex = create_new_vertex(vertex, generator)
                outer_rings.append(new_vertex)
  
    # Next, we create more faces by applying more elements of the Fuchsian group to the unit cell. The number of new faces is the length of the extra_generators_indices list.
    extra_generators_indices = []   
    if p_B == 8:
        if N == 12:
            extra_generators_indices = [(1,2),(0,3),(2,2)]
        if N == 16:
            extra_generators_indices = [(5,2),(2,0),(3,0),(4,0),(5,0),(6,0), (7,0)]
            

    for index_pair in extra_generators_indices:
        # For each pair of indices, we generate an element of the Fuchsian group by matrix multiplication.
        fuchsian_group_element = group_generators[index_pair[0]] @ group_generators[index_pair[1]]
        # Again, we account for duplication for these lattices.
        if not redundant_indices:
            for vertex in unit_cell:
                new_vertex = create_new_vertex(vertex, fuchsian_group_element)
                outer_rings.append(new_vertex)
        else: 
            for vertex in unit_cell:
                new_vertex = create_new_vertex(vertex, fuchsian_group_element)
                i += 1
                if all(hyperbolic_distance(new_vertex, vertex) > (d0 - 0.2) for vertex in unit_cell + outer_rings):
                    outer_rings.append(new_vertex)
                else:
                    outer_rings.append(new_vertex)
                    redundant_indices.append(i)
            
            
    final_vertices = unit_cell + outer_rings
    
    return final_vertices, redundant_indices

# This function takes as input a list of vertices positions and a list of the indices of redundant vertices. It returns the following:
# 1) The graph G constructed from these vertices.
# 2) The adjacency matrix adj_G of G.
# 3) A dictionary vertices_to_edges mapping vertices to the edges connecting them.
# 4) A dictionary edges_to_vertices mapping edges to the vertices at their endpoints.
# 5) A dictionary pos_dict mapping each vertex in the graph to its position.
def generate_hyperbolic_graph(vertices: list, redundant_indices: list, draw=False):
    
    # This dictionary stores the labels of all the edges in the graph.
    vertices_to_edges = {}
    
    # This dictionary stores the two vertices at the endpoints of each edge in the graph
    edges_to_vertices = {}
    
    # This is the distance between nearest neighbours in the hyperbolic lattice.
    d0 = hyperbolic_distance(vertices[0], vertices[1])
    
    # This list stores the coordinates of every vertex in the graph (including redundant vertices).
    coords = [(v.real, v.imag) for v in vertices]
    
    # Initialize the graph that mimics the hyperbolic lattice.
    G = nx.Graph()
    
    # Add nodes to the graph with labels and positions.
    if not redundant_indices:
        for idx, pos in enumerate(coords):
            G.add_node(idx, pos=pos, label=True)
    else:
        for idx, pos in enumerate(coords):
            if idx not in redundant_indices:
                G.add_node(idx, pos=pos, label=True)

    edge_count = 0  # Counter for edge labels
    
    # This is the total number of vertices in the hyperbolic lattice.
    n = len(vertices)
    
    # Initialize the adjacency matrix of the graph
    Adj_G = np.zeros((n,n))
    # Add edges to the graph and construct the adjacency matrix.
    for i, pos1 in enumerate(vertices):
        for j, pos2 in enumerate(vertices[i + 1:], start=i + 1):
            if not redundant_indices:
                if hyperbolic_distance(pos1, pos2) < (d0 + 0.1):
                    Adj_G[i][j] = 1
                    Adj_G[j][i] = 1
                    G.add_edge(i, j, with_labels=True)
                    vertices_to_edges[(i, j)] = edge_count
                    edges_to_vertices[edge_count] = (i, j)
                    edge_count += 1
            else:
                if i not in redundant_indices and j not in redundant_indices:
                    if hyperbolic_distance(pos1, pos2) < (d0 + 0.1):
                        Adj_G[i][j] = 1
                        Adj_G[j][i] = 1
                        G.add_edge(i, j, with_labels=True)
                        vertices_to_edges[(i, j)] = edge_count
                        edges_to_vertices[edge_count] = (i, j)
                        edge_count += 1


    # This dictionary stores the indices and positions of all the vertices in the graph
    pos_dict = {idx: (pos.real,pos.imag) for idx, pos in enumerate(vertices) if idx not in redundant_indices}
    

    # if draw is True, draw the hyperbolic lattice with positions and labels.
    if draw:
        plt.figure(figsize=(30, 30))  # Set figure size
        # Draw the graph
        nx.draw(
            G,
            pos=pos_dict,  # Add node positions
            node_size=20,  # Adjust node size
            node_color="lightblue",
            with_labels=True,
            font_size=12,  # Adjust font size for node labels
            font_color="black"
        )

        # Draw edge labels
        nx.draw_networkx_edge_labels(
            G,
            pos=pos_dict,
            edge_labels=vertices_to_edges,
            font_size=12,  # Adjust font size for edge labels
            label_pos=0.5,  # Adjust edge label position (closer to the center of edges)
        )

        plt.axis("equal")  # Ensure equal scaling
        plt.show()  # Display the plot

    return G, Adj_G, vertices_to_edges, edges_to_vertices, pos_dict

def get_edge_labels_for_vertex(G: nx.Graph, vertex: int, vertices_to_edges: dict):
    """Takes as input a graph and a vertex label and returns the labels of all edges incident on this node."""
    incident_edges = list(G.edges(vertex))
    incident_edge_labels = {edge: vertices_to_edges[tuple(sorted(edge))] for edge in incident_edges}
    return incident_edge_labels.values()

def get_edge_from_v1_v2(v1: int, v2: int, vertices_to_edges: dict):
    """Takes labels for two vertices for an edge and returns the corresponding edge label.
    Edge labels are easier to work and debug with."""
    tup = tuple(sorted((v1, v2)))
    return vertices_to_edges[tup]

def create_adjacency_matrix(adjacency_matrix: np.array, redundant_indices: list, N: int, CT: list, p_B: int, p: int, q: int):
    # Get a list of positions of vertices in the unit cell.
    unit_cell = unit_cell_positions(p, q)
    n = len(unit_cell) 

    # We use two different approaches to create the adjacency matrix of the periodic graph.
    # The first approach when there is redundancy in the vertices.
    # The second approach when there is no redundancy in the vertices.
    if not redundant_indices:
        unit_cell_graph = generate_hyperbolic_graph(unit_cell, redundant_indices)
        # Adjacency matrix of the unit cell
        V = nx.to_numpy_array(unit_cell_graph[0])  
        I = np.identity(N)
        A_l = np.kron(I, V)
    else:
        A_l = adjacency_matrix
        
    # Initialize the inter-cell matrices. These matrices dictate how to glue different unit cells together in the graph.
    T_matrices = [np.zeros((n, n)) for _ in range(p_B)]

    # Indices to be updated

    T_indices = [
        [[9, 12], [8, 13]],
        [[12, 9], [13, 8]],
        [[9, 14], [10, 13]],
        [[14, 9], [13, 10]],
        [[10, 15], [11, 14]],
        [[15, 10], [14, 11]],
        [[11, 8], [12, 15]],
        [[8, 11], [15, 12]]]
        

    # Update the T matrices
    for j in range(p_B):
        for k, l in T_indices[j]:
            T_matrices[j][k, l] = 1

    # Perform the update to A_l based on CT
    if not redundant_indices:
        # Perform the update to A_l based on CT
        for alpha in range(p_B):  # Iterate over the T_matrices
            for i in range(N):
                # Create U matrix
                U = np.zeros((N, N))
                j = CT[alpha][i] - 1  # Adjust indexing for Python (0-based)
                U[i, j] = 1
                # Update A_l
                A_l += np.kron(U, T_matrices[alpha])
    else: 
        # Perform the update to A_l based on CT
        for alpha in range(p_B):  # Iterate over the T_matrices
            for i in range(N):
                # Create U matrix
                U = np.zeros((N, N))
                j = CT[alpha][i] - 1  # Adjust indexing for Python (0-based)
                if i > 0 and  j > 0:
                    U[i, j] = 1
                    # Update A_l
                    A_l += np.kron(U, T_matrices[alpha])
                

    # Convert A_l to a sparse matrix. Do we have to do this step or can we just convert it into COO right away.?
    A_l_sparse = csr_matrix(A_l)

    # Convert to COO format to access row, col, and data attributes
    A_l_coo = coo_matrix(A_l_sparse)
    
    if not redundant_indices:
        sparse_matrix = {(i, j): v for i, j, v in zip(A_l_coo.row, A_l_coo.col, A_l_coo.data)}
    else:
        # Step 1: Remove redundant indices from sparse matrix
        initial_sparse_matrix = {(i, j): v for i, j, v in zip(A_l_coo.row, A_l_coo.col, A_l_coo.data) if i not in redundant_indices and j not in redundant_indices}
        
        # Step 2: Count occurrences of each vertex
        vertex_count = Counter()
        for u, v in initial_sparse_matrix.keys():
            vertex_count[u] += 1
            vertex_count[v] += 1
        
        # Step 3: Find redundant nodes
        redundant_nodes = [node for node, count in vertex_count.items() if count != 2*q]
        
        # Step 4: Create a mapping for redundant nodes
        redundant_mapping = {}
        for i in range(0, len(redundant_nodes) - 1, 2):  
            u, v = redundant_nodes[i], redundant_nodes[i+1]
            redundant_mapping[v] = u  # Map v → u
            
        # Step 5: Update the sparse matrix without modifying non-redundant edges
        sparse_matrix = {}
        for (i, j), v in initial_sparse_matrix.items():
            # If neither i nor j are redundant, keep the edge unchanged
            if i not in redundant_mapping and j not in redundant_mapping:
                sparse_matrix[(i, j)] = v
            elif i in redundant_nodes and j in redundant_nodes:
                continue
            else:
                # Replace only redundant nodes with their mapped values
                new_i = redundant_mapping.get(i, i)
                new_j = redundant_mapping.get(j, j)
                sparse_matrix[(new_i, new_j)] = v  # Preserve edge direction
    
    
    for (i,j), v in sparse_matrix.items():
        if (j,i) not in sparse_matrix:
            raise ValueError("The sparse matrix is not symmetric")
        
    # Step 2: Count occurrences of each vertex
    vertex_count = Counter()
    for u, v in sparse_matrix.keys():
        vertex_count[u] += 1
        vertex_count[v] += 1

    # Step 3: Find redundant nodes
    degree_check = all(count == 2*q for node, count in vertex_count.items())
    if not degree_check:
        print([(node, count) for node,count in vertex_count.items() if count != 2*q])
        raise ValueError("Some nodes do not appear 8 times in the sparse matrix")
    # else:
    #     print(sparse_matrix)
        
    return sparse_matrix

def add_periodicity_edges(original_graph: nx.Graph, vertices_to_edges: dict, edges_to_vertices: dict, sparse_matrix: dict, vertices_positions: dict, draw=False):
    """
    Finds all the new edges based on periodic boundary condition, then adds those edges to a copy of the original graph.
    1. Check if every nonzero element in the adjacency matrix of G is also in the sparse matrix file.
    2. If all edges in G exist in the file, add missing edges from the adjacency matrix file to G.
    """

    periodic_G = copy.deepcopy(original_graph)

    # Step 2: Add edges from the sparse matrix file to G if they are not present in G
    edges_added = []
    edge_count = len(periodic_G.edges())
    for (i, j) in sparse_matrix.keys():
        if not periodic_G.has_edge(i, j):  # If the edge is not in G, add it
            if i in periodic_G.nodes and j in periodic_G.nodes:
                periodic_G.add_edge(int(i), int(j), with_labels = True)
                vertices_to_edges[(i, j)] = edge_count
                edges_to_vertices[edge_count] = (i, j)
                edges_added.append((i, j))
                edge_count += 1
            else:
                raise ValueError(f"Could not add edge; nodes ({i}, {j}) not in graph")

    if draw:
        plt.figure(figsize=(20, 20))
        nx.draw(
            periodic_G,
            pos=vertices_positions,
            node_size=20,
            node_color="lightblue",
            with_labels=True,
            font_size=6,
            font_color="black"
        )
        nx.draw_networkx_edge_labels(
            periodic_G,
            pos=vertices_positions,
            edge_labels=vertices_to_edges,
            font_size=10,
            label_pos=0.5,
        )
    return periodic_G

def generate_bit_flip_circuit(graph: nx.Graph, error_prob: int, vertices_to_edges: dict):
    """Takes the graph as input and generates the corresponding quantum circuit for bit flip error in Qiskit. 
    The bit flip circuit applies a CX between each edge qubit and its incident vertices. 
    A random X error is introduced randomly based on input error probability."""

    affected_edges = []
    num_vertices = len(graph.nodes())
    num_edges = len(graph.edges())

    vertices_qubits = QuantumRegister(num_vertices, 'vertices_qubits')
    edges_qubits = QuantumRegister(num_edges, 'edges_qubits')
    cr = ClassicalRegister(num_vertices, 'cr')
    qc = QuantumCircuit()
    qc.add_register(vertices_qubits)
    qc.add_register(edges_qubits)
    qc.add_register(cr)

    for j in range(len(edges_qubits)):
        random_number = np.random.rand()
        if random_number < error_prob:
            qc.x(edges_qubits[j])
            affected_edges.append(j)

    for v in range(len(graph.nodes())):
        incident_edges = get_edge_labels_for_vertex(graph, v, vertices_to_edges)
        for edge in incident_edges:
            qc.cx(edges_qubits[edge], vertices_qubits[v])

        qc.barrier()

    for v in range(len(graph.nodes())):
        qc.measure(vertices_qubits[v], cr[v])

    return qc, affected_edges

def hyperbolic_cycle_basis(original_graph: nx.Graph, periodic_graph: nx.Graph, p: int, q: int):
    """Based on the Networkx implentation of minimum_cycle_basis. Find a cycle basis for the periodic graph
    to ensure all cycles and logical operators are found."""

    cycle_basis = []
    all_faces = []
    all_nodes = [node for node, degree in original_graph.degree() if degree < q] 
    # We extract the edges not in a spanning tree. We do not really need a
    # *minimum* spanning tree. That is why we call the next function with
    # weight=None. Depending on implementation, it may be faster as well
    tree_edges = list(nx.minimum_spanning_edges(periodic_graph, data = False))
    chords = periodic_graph.edges - tree_edges - {(v, u) for u, v in tree_edges}
    
    # We maintain a set of vectors orthogonal to sofar found cycles
    # Recall that every cycle C (and every witness S) is given as a vector in the chords, that is every cycle (and every witness S) is given as a vector v \in {0,1}^N
    set_orth = [{tuple(sorted(edge))} for edge in chords]
    print(f"The initial length of the set_orth is {len(set_orth)}")
    nx_cycle_basis = nx.minimum_cycle_basis(original_graph)
    original_cycles_length = len(nx_cycle_basis)
    
    # Step 1: Extracting plaquettes from the original graph.
    while len(cycle_basis) < original_cycles_length:
        for i, base in enumerate(set_orth):
            valid_cycles, cycle_edges = valid_cycle_basis(nx_cycle_basis[0], base, p)
            if valid_cycles:
                cycle_basis.append(cycle_edges)
                all_faces.append(cycle_edges) 
                nx_cycle_basis.remove(nx_cycle_basis[0])
                set_orth.remove(base)
                # Update the set of vectors orthogonal to sofar found cycles
                set_orth = [
                    ({e for e in orth if e not in base } | {e for e in base if e not in orth})
                        if sum((e in orth) for e in cycle_edges) % 2
                    
                    else orth
                    for orth in set_orth
                            ]
                break
    
    print(f"The length of remaining cycles from the original graph is {len(nx_cycle_basis)}")    

    num_plaquettes = periodic_graph.number_of_edges()*2/p
    print("num_plaquettes", num_plaquettes)
    
    print("len(set_orth) before p_cycles", len(set_orth))
   # Step 2: Extracting extra plaquettes that were added due to imposing periodic boundary conditions.
    while len(all_faces) < num_plaquettes:
        cycle_edges = p_cycles(periodic_graph, set_orth, cycle_basis, num_plaquettes, all_nodes, p)
        if cycle_edges:
            all_faces.append(cycle_edges)
            if len(cycle_basis) < num_plaquettes-1:
                cycle_basis.append(cycle_edges)
    print("len of cycle_basis after p_cycles", len(cycle_basis))
    
    # Step 3: Find all non_trivial cycles (which form the logical operators)
    logical_operators = []
    first_logical_operator = first_non_trivial_cycle_(periodic_graph,set_orth,all_faces)
    logical_operators.append(first_logical_operator)
    cycle_basis.append(first_logical_operator)


    first_operator_vertices = set()
    for u, v in first_logical_operator:
        first_operator_vertices.add(u)
        first_operator_vertices.add(v)

    print("len(set_orth) before non_trivial", len(set_orth))
    for base_point in first_operator_vertices:
        set_orth_copy = copy.deepcopy(set_orth)
        found_logical_operators = []
        while set_orth_copy:
            cycle_edges, base_point_error = non_trivial_cycles(periodic_graph, set_orth_copy, all_faces, base_point)
            if base_point_error:
                break
            found_logical_operators.append(cycle_edges)
        if len(set_orth_copy) == 0:
            break
    set_orth = set_orth_copy
    logical_operators.extend(found_logical_operators)
    cycle_basis.extend(found_logical_operators)
    print("len(logical_operators)", len(logical_operators))
        
    print("length of logical_op after non_trivia_cycles", len(logical_operators))


    if set_orth:
        raise ValueError("The number of cycles in the cycle basis is not E-V+1")
    print(f"The number of cycles in total is {len(cycle_basis)}")
    print(f"The remaining number of cycles in the basis is {len(set_orth)}")
    non_trivial_cycles_ = [cycle for cycle in cycle_basis if cycle not in all_faces]
    return cycle_basis, all_faces, logical_operators      
            
def p_cycles(periodic_graph: nx.Graph, set_orth: list, cycle_basis: list, num_plaquettes: int, all_nodes: list, p: int):    
    
    for base in set_orth:
        # Add 2 copies of each edge in G to Gi.
        # If edge is in orth, add cross edge; otherwise in-plane edge
        Gi = nx.Graph()
        for u, v in periodic_graph.edges():
            if (u, v) in base or (v, u) in base:
                Gi.add_edges_from([(u, (v, 1)), ((u, 1), v)],)
            else:
                Gi.add_edges_from([(u, v), ((u, 1), (v, 1))])
        
            
        chosen_plaquette = None
        found = False
        for node in all_nodes:
            start = node
            end = (start,1) 
            potential_plaquettes = find_unique_paths_dfs(Gi, start, end, p)
            
            if potential_plaquettes:
                for plaquette in potential_plaquettes:
                    # We need to check that this condiiton is valid (plaquette and cycle_basis have same type)
                    if plaquette not in cycle_basis:
                        # Check that this notation is correct
                        valid_plaquette = all(len(plaquette.intersection(cycle))<= 1 for cycle in cycle_basis)
                        if valid_plaquette:
                            chosen_plaquette = plaquette
                            found = True
                            break

            if found and len(cycle_basis) < num_plaquettes-1:
                set_orth.remove(base)
                set_orth = [
                    ({e for e in orth if e not in base } | {e for e in base if e not in orth})
                     if sum((e in orth) for e in chosen_plaquette) % 2
                    
                    else orth
                    for orth in set_orth
                            ]
                break
        if found:
            break
    
    return chosen_plaquette   
    
def valid_cycle_basis(cycle: list, S: dict, p: int):
    """
    This function a cycle from nx.minimum_cycle_basis and checks if it is a valid cycle for a given base S
    by checking the condition that <C,S> = 1.
    """
    # First convert the list cycle to a set of edges.
    plaquette_edges = OrderedSet()

    for i in range(len(cycle)):
        u, v = cycle[i], cycle[(i + 1) %p]
        plaquette_edges.add(tuple(sorted((u, v)))) 

    common_edges = plaquette_edges.intersection(S)

    # Check the condition <C,S> = 1.
    valid_cycle = len(common_edges) % 2 == 1
    
    return valid_cycle, plaquette_edges

def find_unique_paths_dfs(G: nx.Graph, source: int, target: int, p: int):
    def dfs(node, path, length):
        if length == p:
            if node == target:
                normalized_edges = OrderedSet()
                for i in range(len(path) - 1):  # Avoid self-loops by not wrapping around
                    u, v = path[i], path[i + 1]
                    if type(u) is tuple:
                        u = u[0]
                    if type(v) is tuple: 
                        v = v[0]
                    normalized_edges.add(tuple(sorted((u, v)))) 
                if normalized_edges not in paths_list:
                    paths_list.append(normalized_edges)
            return

        for neighbor in G.neighbors(node):
            if neighbor in path and neighbor != target:  # Avoid revisiting nodes unless it's the target
                continue
            dfs(neighbor, path + [neighbor], length + 1)

    paths_list = []

    dfs(source, [source], 0)

    return paths_list

def first_non_trivial_cycle_(G: nx.Graph, set_orth: list, all_faces: list):
    """This function is the same as non_trivial_cycles, but it stops after finding the first non_trivial_cycle
    and returns a base point to be used to find the remaining non-trivial cycles, since they all should pass by a base point."""
    Gi = nx.Graph()
    
    for base in set_orth:
        # Add 2 copies of each edge in G to Gi.
        # If edge is in orth, add cross edge; otherwise in-plane edge
        for u, v in G.edges():
            if (u, v) in base or (v, u) in base:
                Gi.add_edges_from([(u, (v, 1)), ((u, 1), v)])
            else:
                Gi.add_edges_from([(u, v), ((u, 1), (v, 1))])
    
        # find the shortest length in Gi between n and (n, 1) for each n
        # Note: Use "Gi_weight" for name of weight attribute
        spl = nx.shortest_path_length
        lift = {node: spl(Gi, source=node, target=(node, 1)) for node in G}
        sorted_lift = dict(sorted(lift.items(), key=lambda item: item[1]))
        
        
        found = False
        for node, length in sorted_lift.items():
            start = node
            end = (start, 1)
            potential_cycles = find_unique_paths_dfs(Gi, start, end, length)
            for potential_path in potential_cycles:
                if potential_path not in all_faces:
                    non_trivial_cycle = potential_path
                    set_orth.remove(base)
                    set_orth = [
                            ({e for e in orth if e not in base } | {e for e in base if e not in orth})
                                if sum((e in orth) for e in non_trivial_cycle) % 2
                            
                            else orth
                            for orth in set_orth
                                    ]
                    found = True
                    break
            if found:
                break
        if found:
            break
    
    return non_trivial_cycle

def non_trivial_cycles(G: nx.Graph, set_orth: list, all_faces: list, base_point: int):
    """
    Computes the minimum weight cycle in G,
    orthogonal to the vector orth as per [p. 338, 1]
    Use (u, 1) to indicate the lifted copy of u (denoted u' in paper).
    """
    
    Gi = nx.Graph()

    found = False
    error = False
    for base in set_orth:
        # Add 2 copies of each edge in G to Gi.
        # If edge is in orth, add cross edge; otherwise in-plane edge
        for u, v in G.edges():
            if (u, v) in base or (v, u) in base:
                Gi.add_edges_from([(u, (v, 1)), ((u, 1), v)])
            else:
                Gi.add_edges_from([(u, v), ((u, 1), (v, 1))])

        start = base_point
        end = (start, 1)
        length = nx.shortest_path_length(Gi, start, end)
        potential_cycles = find_unique_paths_dfs(Gi, start, end, length)
        non_trivial_cycle = None
        for potential_path in potential_cycles:
            # Should we also put the condition if potential_path not in cycle_basis to avoid double counting?
            if potential_path not in all_faces:
                non_trivial_cycle = potential_path
                set_orth.remove(base)
                set_orth = [
                        ({e for e in orth if e not in base } | {e for e in base if e not in orth})
                        if sum((e in orth) for e in non_trivial_cycle) % 2
                        
                        else orth
                        for orth in set_orth
                                ]
                found = True
                break
        if found:
            break
    if not found:
        error = True
    
    return non_trivial_cycle, error

def logical_operators_to_edges(logical_operators: list, vertices_to_edges: dict):
    """Convert logical operators from a list of pairs of vertices (v1,v2) to edge labels."""
    logical_operators_edges = []

    for logical_operator in logical_operators:
        lg_edges = set()
        for pair in logical_operator:
            lg_edges.add(
            get_edge_from_v1_v2(pair[0], pair[1], vertices_to_edges))
        logical_operators_edges.append(lg_edges)
    return logical_operators_edges

def generate_dual_graph(all_faces: list, p: int, vertices_to_edges: dict, G_pos_dict: dict, draw=False):
    """Generate the dual graph which replaces each plaquette with a vertix. Used for computing the phase flip circuit."""
    G_dual_graph = nx.Graph()


    pos_dict = {}

    # For each face in graph, add a node to dual graph with its position at the average of all the face's nodes
    for i, plaquette in enumerate(all_faces):
        plaquette_vertices = set()
        for pair in plaquette:
            plaquette_vertices.add(pair[0])
            plaquette_vertices.add(pair[1])
        
        x_pos = 0.0
        y_pos = 0.0
        for v in plaquette_vertices:
            x_pos += G_pos_dict[v][0]
            y_pos += G_pos_dict[v][1]
        x_pos = x_pos/p
        y_pos = y_pos/p
        pos = (x_pos, y_pos)
        pos_dict[i] = pos

        G_dual_graph.add_node(i, pos=pos, label = True)
        
    intersection_edges = {}
    
    for i, cycle1 in enumerate(all_faces):
        for j, cycle2 in enumerate(all_faces[i+1:], start=i+1):
            intersection = list(cycle1.intersection(cycle2))
            if len(intersection) == 1:
                label = get_edge_from_v1_v2(intersection[0][0],intersection[0][1], vertices_to_edges)
                G_dual_graph.add_edge(i, j, label=label)
                intersection_edges[(i,j)] = label
            elif len(intersection) == 0:
                continue
            else:
                raise ValueError(f"The intersection between two faces {i} and {j} is {len(cycle1.intersection(cycle2))}")


    if draw:
        plt.figure(figsize=(10, 10)) 
       
        nx.draw(
            G_dual_graph,
            pos=pos_dict,
            node_size=20, 
            node_color="lightblue",
            with_labels=True,
            font_size=12, 
            font_color="black"
            )
        nx.draw_networkx_edge_labels(
            G_dual_graph,
            pos=pos_dict,
            edge_labels=intersection_edges,
            font_size=10,
            label_pos=0.5,
        )
        plt.show()
        
    return G_dual_graph, intersection_edges

def generate_phase_flip_circuit(periodic_G: nx.Graph, all_faces: list, graph_vertices_to_edges: dict, err_prob: float):
    affected_edges = []
    num_plaquettes = len(all_faces)
    print('num_vertices', num_plaquettes)
    num_edges = periodic_G.number_of_edges()
    print('num_edges', num_edges)

    plaquette_ancilla_qubits = QuantumRegister(num_plaquettes, 'plaquette_ancilla_qubits')
    edges_qubits = QuantumRegister(num_edges, 'edges_qubits')
    cr = ClassicalRegister(num_plaquettes, 'cr')

    qc = QuantumCircuit()
    qc.add_register(plaquette_ancilla_qubits)
    qc.add_register(edges_qubits)
    qc.add_register(cr)

    # H on all qubits to change basis
    for i in range(num_plaquettes):
        qc.h(plaquette_ancilla_qubits[i])
    for j in range(num_edges):
        qc.h(edges_qubits[j])
    qc.barrier()

    # add random Z errors
    for j in range(num_edges):
        random_number = np.random.rand()
        if random_number < err_prob:
            qc.z(edges_qubits[j])
            affected_edges.append(j)
    qc.barrier()
    

    # CX between centre ancilla and neighboring edges
    for i, plaquette in enumerate(all_faces):
        for edge in plaquette:
            edge_label = get_edge_from_v1_v2(edge[0], edge[1], graph_vertices_to_edges)
            qc.cx(plaquette_ancilla_qubits[i], edges_qubits[edge_label])
        qc.barrier()
    qc.barrier()
        
    # H on all again
    for i in range(num_plaquettes):
        qc.h(plaquette_ancilla_qubits[i])
    for j in range(num_edges):
        qc.h(edges_qubits[j])
    qc.barrier()

    # Z measurement
    for v in range(num_plaquettes):
        qc.measure(plaquette_ancilla_qubits[v], cr[v])

    return qc, affected_edges

def generate_syndrome_graph(indices: list, graph: nx.Graph, draw=False):
    syndrome_graph = nx.Graph()
    for v in indices:
        syndrome_graph.add_node(v, label = True)

    for i, ver1 in enumerate(indices):
        for j, ver2 in enumerate(indices[i+1:], start=i+1):
            weight = nx.shortest_path_length(graph, source=ver1, target=ver2)
            syndrome_graph.add_edge(ver1, ver2, weight=weight)

    # Draw the syndrome graph
    if draw:
        pos = nx.spring_layout(syndrome_graph)  # Positioning for better visualization
        nx.draw(syndrome_graph, pos, with_labels=True, node_color="lightblue", node_size=500)

        # Add edge labels showing the weights
        edge_labels_with_weight = nx.get_edge_attributes(syndrome_graph, 'weight')  # Retrieve weights
        nx.draw_networkx_edge_labels(syndrome_graph, pos, edge_labels=edge_labels_with_weight)
    
        # Show the graph
        plt.show()
    return syndrome_graph

def find_correction_paths(syndrome_graph: nx.Graph, periodic_graph: nx.Graph):
    # Find the MWPM from the syndrome graph because edges are weights, 
    # then find the actual correction path using the periodic graph because syndrom graph doesnt contain all nodes
    matching = nx.algorithms.matching.min_weight_matching(syndrome_graph)

    correction_paths = []
    
    for edge in matching:
        path = nx.shortest_path(periodic_graph, source=edge[0], target=edge[1])
        correction_paths.append(path)
    return correction_paths

def get_logical_error(affected_edges: list, correction_paths: list, vertices_to_edges: dict, logical_operators: list):
    
    logical_operators_edges = logical_operators_to_edges(logical_operators, vertices_to_edges)
    
    affected_edges_set = set(affected_edges)
    all_correction_edges = set()
    for correction_path in correction_paths:
        for i in range(len(correction_path) - 1):
            all_correction_edges.add(get_edge_from_v1_v2(correction_path[i], correction_path[i + 1], vertices_to_edges))

    if all_correction_edges == affected_edges_set:
        return False

    # remove the common edges between affected and correction edges
    intersection = all_correction_edges.intersection(affected_edges_set)
    union = all_correction_edges.union(affected_edges_set)
    potential_logical_error = union - intersection
    
    for logical_operator in logical_operators_edges:
        common_edges = potential_logical_error.intersection(logical_operator)
        if len(common_edges) % 2 == 1:
            return True
    
    return False

def run_trial(args):
    periodic_graph, vertices_to_edges, ep, logical_operators = args
    qcc, affected_edges = generate_bit_flip_circuit(periodic_graph, ep, vertices_to_edges)
    simulator = AerSimulator()
    result = simulator.run(qcc, shots=1).result().get_counts()
    syndrome_measurement = next(iter(result))[::-1]
    indices = [j for j, bit in enumerate(syndrome_measurement) if bit == "1"]
    syndrome_graph = generate_syndrome_graph(indices, periodic_graph)
    correction_paths = find_correction_paths(syndrome_graph, periodic_graph)
    is_err = get_logical_error(affected_edges, correction_paths, vertices_to_edges, logical_operators)
    return is_err

def error_graph(periodic_graph, vertices_to_edges, error_probabilities, logical_operators):
    trials = 5000
    error_percentages = []
    for ep in error_probabilities:
        print(f"Processing error probability {ep} for the graph with {periodic_graph.number_of_edges()} qubits")
        args_list = [(periodic_graph, vertices_to_edges, ep, logical_operators) for _ in range(trials)]
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(run_trial, args_list)
        count = sum(results)
        error_percentages.append(count / trials)
    return error_percentages

def find_code_distance(logical_operators, periodic_graph):
    distance = 100
    for logical_operator in logical_operators:
        Gi = nx.Graph()
        operator_len = 100
        operator_vertices = set()
        for u, v in periodic_graph.edges():
            operator_vertices.add(u)
            operator_vertices.add(v)
            if (u, v) in logical_operator or (v, u) in logical_operator:
                Gi.add_edges_from([(u, (v, 1)), ((u, 1), v)])
            else:
                Gi.add_edges_from([(u, v), ((u, 1), (v, 1))])
        
        for v in operator_vertices:
            start = v
            end = (start, 1)
            length = nx.shortest_path_length(Gi, start, end)
            path = nx.shortest_path(Gi, start, end)
            if length < operator_len:
                operator_len = length
        
        if operator_len < distance:
            distance = operator_len
    return distance
if __name__ == '__main__':
    p = 8
    q = 3
    p_B = 8
    q_B = 8
    
    if p_B == 8:
        CT_matrices_dic = {                
            # Abelian Subgroup, NSG[365]
            9: [ [ 2, 3, 1, 6, 8, 9, 5, 7, 4 ], 
                 [ 3, 1, 2, 9, 7, 4, 8, 5, 6 ], 
                 [ 4, 6, 9, 5, 1, 8, 3, 2, 7 ], 
                 [ 5, 8, 7, 1, 4, 2, 9, 6, 3 ], 
                 [ 6, 9, 4, 8, 2, 7, 1, 3, 5 ], 
                 [ 7, 5, 8, 3, 9, 1, 6, 4, 2 ], 
                 [ 8, 7, 5, 2, 6, 3, 4, 9, 1 ], 
                 [ 9, 4, 6, 7, 3, 5, 2, 1, 8 ] ],
            
            
            # Abelian Subgroup, NSG[2872]
            12: [ [ 2, 10, 1, 6, 11, 9, 5, 7, 12, 4, 3, 8 ], 
                  [ 3, 1, 11, 10, 7, 4, 8, 12, 6, 2, 5, 9 ], 
                  [ 4, 6, 10, 12, 1, 8, 3, 11, 7, 9, 2, 5 ], 
                  [ 5, 11, 7, 1, 12, 2, 9, 6, 10, 3, 8, 4 ], 
                  [ 6, 9, 4, 8, 2, 7, 1, 3, 5, 12, 10, 11 ], 
                  [ 7, 5, 8, 3, 9, 1, 6, 4, 2, 11, 12, 10 ], 
                  [ 8, 7, 12, 11, 6, 3, 4, 10, 1, 5, 9, 2 ], 
                  [ 9, 12, 6, 7, 10, 5, 2, 1, 11, 8, 4, 3 ] ],

                    
#             # Abelian Subgroup, NSG[10425]
            16: [ [ 2, 10, 1, 11, 12, 13, 14, 15, 16, 3, 6, 7, 4, 5, 9, 8 ], 
                  [ 3, 1, 10, 13, 14, 11, 12, 16, 15, 2, 4, 5, 6, 7, 8, 9 ], 
                  [ 4, 11, 13, 8, 1, 9, 10, 7, 5, 6, 15, 2, 16, 3, 14, 12 ], 
                  [ 5, 12, 14, 1, 9, 10, 8, 4, 6, 7, 2, 16, 3, 15, 11, 13 ], 
                  [ 6, 13, 11, 9, 10, 8, 1, 5, 7, 4, 16, 3, 15, 2, 12, 14 ], 
                  [ 7, 14, 12, 10, 8, 1, 9, 6, 4, 5, 3, 15, 2, 16, 13, 11 ], 
                  [ 8, 15, 16, 7, 4, 5, 6, 10, 1, 9, 14, 11, 12, 13, 3, 2 ], 
                  [ 9, 16, 15, 5, 6, 7, 4, 1, 10, 8, 12, 13, 14, 11, 2, 3 ] ]
            }
        
    
    plt.figure(figsize=(10, 10)) 

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    error_probabilities = [0.05 * j for j in range(11)]
    for idx, N in enumerate(CT_matrices_dic):
        G_vertices, redundant_vertices = generate_vertices(p, q, p_B, q_B, N)
        
        G, adj_G, G_vertices_to_edges, G_edges_to_vertices, G_pos_dict = generate_hyperbolic_graph(G_vertices, redundant_vertices, draw=False)
        
        sparse_matrix = create_adjacency_matrix(adj_G, redundant_vertices, N, CT_matrices_dic[N], p_B, p, q)
        
        periodic_G = add_periodicity_edges(G, G_vertices_to_edges, G_edges_to_vertices, sparse_matrix, G_pos_dict)
        
        HCB, all_faces, logical_operators = hyperbolic_cycle_basis(G, periodic_G, p, q)

        n = periodic_G.number_of_edges()
        k = 2 * (N + 1)
        encoding_rate = k / n
        d = find_code_distance(logical_operators, periodic_G)
        
        error_percentage = error_graph(periodic_G, G_vertices_to_edges, error_probabilities, logical_operators)
        
        plt.plot(error_probabilities, error_percentage, marker='o', linestyle='-',
                 color=colors[idx % len(colors)], label=f'[[n={n}, k={k}, d={d}]]')


    plt.xlabel('Physical Error Probabilities')
    plt.ylabel('Logical Error Probabilities')
    plt.title('Error Threshold Graph')
    plt.legend()
    plt.grid(True)
    end = time.time()
    print(end - start)
    plt.show() 

