import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\processesd_data\final_competition_cases.csv")
df = df.dropna(subset=["related_to"])

G = nx.DiGraph() 

for _, row in df.iterrows():
    source = row["article_id"]
    targets = [t.strip() for t in str(row["related_to"]).split(";")]
    for target in targets:
        if target:
            G.add_edge(source, target)
            # carry over attributes for source node
            G.nodes[source]["sector"] = row.get("sector", "")
            G.nodes[source]["year"]   = row.get("year", "")
            G.nodes[source]["stage"]  = row.get("decision_stage", "")

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# Most cited cases (in-degree = how often a case is referenced by others)
in_degree = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
print("\nMost cited cases (referenced by others):")
for node, deg in in_degree[:10]:
    print(f"  {node}: cited {deg} times")

# Cases that cite the most others (out-degree)
out_degree = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)
print("\nCases citing the most others:")
for node, deg in out_degree[:10]:
    print(f"  {node}: references {deg} cases")

# Connected components (clusters of related cases)
undirected = G.to_undirected()
components = sorted(nx.connected_components(undirected), key=len, reverse=True)
print(f"\nConnected components: {len(components)}")
print(f"Largest cluster: {len(components[0])} cases")
print(f"Cases in largest cluster: {list(components[0])[:10]} ...")

largest = G.subgraph(components[0])

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(largest, seed=42, k=1.5)

# Node size = how often it's cited
sizes = [300 + largest.in_degree(n) * 100 for n in largest.nodes()]

nx.draw_networkx_nodes(largest, pos, node_size=sizes, node_color="steelblue", alpha=0.8)
nx.draw_networkx_edges(largest, pos, arrows=True, alpha=0.4, edge_color="gray", arrowsize=10)
nx.draw_networkx_labels(largest, pos, font_size=6)

plt.title("Case Citation Network — largest cluster\n(node size = times cited)")
plt.axis("off")
plt.tight_layout()
plt.savefig(r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\plots\citation_network.png", dpi=120)
plt.show()