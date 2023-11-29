import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


num_null = 2

# Function to calculate rich club coefficient
def calculate_rich_club_coefficient(G):
    return nx.rich_club_coefficient(G, normalized=False)
    

def null_model(G):
    
    H = G.copy()
    
    #HH = nx.configuration_model(list(dict(H.degree()).values()))
    
    HH = nx.random_reference(H)
    
    return HH

# Function to rewire edges to increase rich-club connectivity
def rewire_to_decrease_rich_club(G, num_swaps):
    # Copy the graph to avoid altering the original
    H = G.copy()
    # Perform specified number of edge swaps
    sk = 0
    high = 90
    low = 10
    for _ in range(num_swaps):
        
        # Choose a random high-degree node
        low_degree_nodes = sorted(H.degree, key=lambda x: x[1])[:int(0.1 * len(H.nodes()))]
        low_degree_nodes = [node for node, degree in low_degree_nodes]
        
        u = np.random.choice(low_degree_nodes)
        # Choose random edge of low-degree node that connects to a low-degree node
        possible_edges = [(v) for n, v in H.edges(u) if H.degree[v] < np.percentile(list(dict(H.degree()).values()),low)]
        
        while not possible_edges:
            u = np.random.choice(low_degree_nodes)
            # Choose random edge of low-degree node that connects to a low-degree node
            possible_edges = [(v) for n, v in H.edges(u) if H.degree[v] < np.percentile(list(dict(H.degree()).values()),low)]
            low = low+0.25
        v = np.random.choice(possible_edges)
        
        # Choose a random high-degree node
        high_degree_nodes = sorted(H.degree, key=lambda x: x[1], reverse=True)[:int(0.1 * len(H.nodes()))]
        high_degree_nodes = [node for node, degree in high_degree_nodes]
        x = np.random.choice(high_degree_nodes)
        # Choose random edge of high-degree node that connects to a high-degree node
        possible_edges = [(y) for n, y in H.edges(x) if H.degree[y] > np.percentile(list(dict(H.degree()).values()),high)]
        while not possible_edges:
            x = np.random.choice(high_degree_nodes)
            # Choose random edge of high-degree node that connects to a high-degree node
            possible_edges = [(y) for n, y in H.edges(x) if H.degree[y] > np.percentile(list(dict(H.degree()).values()),high)]
            high = high-0.25
        y = np.random.choice(possible_edges)

        


        # Swap edges if it doesn't duplicate or self-connect
        if (u, y) not in H.edges() and (x, v) not in H.edges() and u != y and x != v:
            H.remove_edge(u, v)
            H.remove_edge(x, y)
            H.add_edge(u, y)
            H.add_edge(x, v)
    return H

# Generate a baseline network
N = 100  # Number of nodes
#E = 2000  # Number of edges
#G = nx.gnm_random_graph(N, E)
G = nx.barabasi_albert_graph(N, 10)

print("num edges: ",G.number_of_edges())

# Calculate the original rich-club coefficient
original_rich_club = calculate_rich_club_coefficient(G)
original_rich_club_nulls = np.zeros((num_null,len(list(original_rich_club.values()))))
print("original null")
for n in range(num_null):
    print(n)
    HH_rewired = null_model(G)
    RCN = calculate_rich_club_coefficient(HH_rewired)
    original_rich_club_nulls[n,:] = list(RCN.values())
# Rewire the network to increase rich-club connectivity
G_rewired = rewire_to_decrease_rich_club(G, num_swaps=900)

# Calculate the new rich-club coefficient
new_rich_club = calculate_rich_club_coefficient(G_rewired)
new_rich_club_nulls = np.zeros((num_null,len(list(new_rich_club.values()))))
print("new null")
for n in range(num_null):
    print(n)
    HH_rewired = null_model(G_rewired)
    RCN = calculate_rich_club_coefficient(HH_rewired)
    new_rich_club_nulls[n,:] = list(RCN.values())

# Plot the rich-club coefficient
plt.figure(figsize=(6, 5))

#plt.subplot(1, 2, 1)
plt.title('Original Rich-Club Coefficient')
plt.plot(list(original_rich_club.values()), label='Original')
plt.xlabel('Degree')
plt.ylabel('Rich-Club Coefficient')
plt.ylim([0, 1.1])


plt.plot(np.mean(original_rich_club_nulls,axis=0), label='null')
plt.xlabel('Degree')
plt.ylabel('Rich-Club Coefficient')
plt.ylim([0, 1.1])

plt.tight_layout()
plt.savefig("figures/original RC Barabasi Albert network.png")
plt.show()

plt.figure(figsize=(6, 5))
#plt.subplot(1, 2, 2)
plt.title('New Rich-Club Coefficient')
plt.plot(list(new_rich_club.values()), label='Rewired')
plt.xlabel('Degree')
plt.ylim([0, 1.1])

plt.plot(np.mean(new_rich_club_nulls,axis=0), label='null')
plt.xlabel('Degree')
plt.ylabel('Rich-Club Coefficient')
plt.ylim([0, 1.1])

plt.tight_layout()
plt.savefig("figures/destroyed RC Barabasi Albert network.png")
plt.show()

RC_network = nx.to_numpy_array(G)

np.save("data/RC_network.npy",RC_network)

no_RC_network = nx.to_numpy_array(G_rewired)

np.save("data/no_RC_network.npy",no_RC_network)

plt.figure()
plt.subplot(1, 2, 1)
plt.title('Rich-Club Network')
plt.imshow(RC_network)

plt.subplot(1, 2, 2)
plt.title('No Rich-Club Network')
plt.imshow(no_RC_network)
plt.tight_layout()
plt.savefig("figures/yes_vs_no_RC_network_matrices.png")
plt.show()