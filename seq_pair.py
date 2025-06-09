from z3 import *
import matplotlib.pyplot as plt
import networkx as nx
import random
blocks = ['A', 'B', 'C', 'D']
trial_input = [
        ([('A','B'),'C','D'],'v'),
        (['A','C'],'vo')
    ]
def seq_pair(blocks,input):
    
    n = len(blocks)

    # Randomize block order
    random.shuffle(blocks)

    # Z3 symbolic variables for positions in each sequence
    pi_plus = {b: Int(f'plus_{b}') for b in blocks}
    pi_minus = {b: Int(f'minus_{b}') for b in blocks}

    s = Solver()

    # Each position must be a valid index
    for b in blocks:
        s.add(And(0 <= pi_plus[b], pi_plus[b] < n))
        s.add(And(0 <= pi_minus[b], pi_minus[b] < n))

    # Ensure valid permutations (distinct positions)
    s.add(Distinct(*pi_plus.values()))
    s.add(Distinct(*pi_minus.values()))
    def horizontal_symmetry(a, b):
        s.add(If(pi_plus[a] < pi_plus[b], pi_minus[a] < pi_minus[b], pi_minus[a] > pi_minus[b]))
    def vertical_symmetry(a, b):
        s.add(If(pi_minus[a] < pi_minus[b], pi_plus[a] > pi_plus[b], pi_plus[a] < pi_plus[b]))

    def enforce_horizontal_order(l):
        for i in range(len(l) - 1):
            a, b = l[i], l[i + 1]
            s.add(And(pi_plus[a] < pi_plus[b], pi_minus[a] < pi_minus[b]))
    def enforce_vertical_order(l):
        for i in range(len(l) - 1):
            a, b = l[i], l[i + 1]
            s.add(And(pi_plus[a] < pi_plus[b] , pi_minus[a] > pi_minus[b]))
    for i in input:
        if i[1] == 'h':
            for j in i[0]:
                if len(j) == 2:
                    horizontal_symmetry(j[0], j[1])
                else:
                    for k in i[0]:
                        if len(k) == 2:
                            s.add(Or(And(And(pi_plus[k[0]] < pi_plus[j[0]], pi_minus[k[0]] < pi_minus[j[0]]),
                                    And(pi_plus[j[0]] < pi_plus[k[1]], pi_minus[j[0]] < pi_minus[k[1]])),
                                    And(And(pi_plus[k[0]] > pi_plus[j[0]], pi_minus[k[0]] > pi_minus[j[0]]),
                                    And(pi_plus[j[0]] > pi_plus[k[1]], pi_minus[j[0]] > pi_minus[k[1]]))))
        if i[1] == 'v':
            for j in i[0]:
                if len(j) == 2:
                    vertical_symmetry(j[0], j[1])
                else:
                    for k in i[0]:
                        if len(k) == 2:
                            s.add(Or(And(And(pi_plus[k[0]] < pi_plus[j[0]], pi_minus[k[0]] > pi_minus[j[0]]),
                                    And(pi_plus[j[0]] < pi_plus[k[1]], pi_minus[j[0]] > pi_minus[k[1]])),
                                    And(And(pi_plus[k[0]] > pi_plus[j[0]], pi_minus[k[0]] < pi_minus[j[0]]),
                                    And(pi_plus[j[0]] > pi_plus[k[1]], pi_minus[j[0]] < pi_minus[k[1]]))))
        if i[1] == 'ho':
            enforce_horizontal_order(i[0])
        if i[1] == 'vo':
            enforce_vertical_order(i[0])
            
    # Add pairwise symmetry logic for all pairs in each group for 'h' and 'v'
    for i in input:
        group, tag = i
        if tag in {'h', 'v'}:
            pairs = [pair for pair in group if isinstance(pair, tuple) and len(pair) == 2]
            for idx1 in range(len(pairs)):
                for idx2 in range(len(pairs)):
                    if idx1 == idx2:
                        continue
                    a, b = pairs[idx1]
                    c, d = pairs[idx2]
                    if tag == 'h':
                        s.add(If(And(pi_plus[a] < pi_plus[c],pi_minus[a] < pi_minus[c]),And(pi_plus[b] > pi_plus[d], pi_minus[b] > pi_minus[d]), True))
                        s.add(If(And(pi_plus[a] > pi_plus[c],pi_minus[a] > pi_minus[c]),And(pi_plus[b] < pi_plus[d], pi_minus[b] < pi_minus[d]), True))
                        s.add(If(And(pi_plus[a] < pi_plus[c],pi_minus[a] > pi_minus[c]),And(pi_plus[b] < pi_plus[d], pi_minus[b] > pi_minus[d]), True))
                        s.add(If(And(pi_plus[a] > pi_plus[c],pi_minus[a] < pi_minus[c]),And(pi_plus[b] > pi_plus[d], pi_minus[b] < pi_minus[d]), True))
                    elif tag == 'v':
                        s.add(If(And(pi_plus[a] < pi_plus[c],pi_minus[a] < pi_minus[c]),And(pi_plus[b] < pi_plus[d], pi_minus[b] < pi_minus[d]), True))
                        s.add(If(And(pi_plus[a] > pi_plus[c],pi_minus[a] > pi_minus[c]),And(pi_plus[b] > pi_plus[d], pi_minus[b] > pi_minus[d]), True))
                        s.add(If(And(pi_plus[a] < pi_plus[c],pi_minus[a] > pi_minus[c]),And(pi_plus[b] > pi_plus[d], pi_minus[b] < pi_minus[d]), True))
                        s.add(If(And(pi_plus[a] > pi_plus[c],pi_minus[a] < pi_minus[c]),And(pi_plus[b] < pi_plus[d], pi_minus[b] > pi_minus[d]), True))
    if s.check() == sat:
        model = s.model()
        pi_plus_list = sorted(blocks, key=lambda b: model[pi_plus[b]].as_long())
        pi_minus_list = sorted(blocks, key=lambda b: model[pi_minus[b]].as_long())
        print("π₊:", pi_plus_list)
        print("π₋:", pi_minus_list)
    else:
        print("No valid sequence pair found under constraints.")
    return model, blocks, pi_plus, pi_minus
model, blocks, pi_plus, pi_minus = seq_pair(blocks,trial_input)


def plot_sequence_pair(m, blocks, pi_plus, pi_minus, realistic=False):
    if realistic:
        sizes = {b: (random.randint(1, 3), random.randint(1, 3)) for b in blocks}
    else:
        sizes = {b: (2, 2) for b in blocks}

    n = len(blocks)
    # Build constraint graphs
    Gx = nx.DiGraph()
    Gy = nx.DiGraph()
    for b in blocks:
        Gx.add_node(b)
        Gy.add_node(b)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b_ = blocks[i], blocks[j]
            ap, bp = m[pi_plus[a]].as_long(), m[pi_plus[b_]].as_long()
            am, bm = m[pi_minus[a]].as_long(), m[pi_minus[b_]].as_long()

            if ap < bp and am < bm:
                Gx.add_edge(a, b_)  # a left of b
            elif ap > bp and am > bm:
                Gx.add_edge(b_, a)  # b left of a (a right of b)
            elif ap < bp and am > bm:
                Gy.add_edge(a, b_)  # a above b
            elif ap > bp and am < bm:
                Gy.add_edge(b_, a)  # b above a (a below b)

    # Assign actual coordinates
    x = {b: 0 for b in blocks}
    y = {b: 0 for b in blocks}
    for node in nx.topological_sort(Gx):
        if Gx.in_degree(node) == 0:
            x[node] = 0
        else:
            x[node] = max(
                x[pred] + sizes[pred][0] + (random.uniform(0.0, 1.0) if realistic else 0)
                for pred in Gx.predecessors(node)
            )
    # Reverse the vertical graph for correct "above" interpretation
    Gy_rev = Gy.reverse()
    for node in nx.topological_sort(Gy_rev):
        if Gy_rev.in_degree(node) == 0:
            y[node] = 0
        else:
            y[node] = max(
                y[pred] + sizes[pred][1] + (random.uniform(0.0, 1.0) if realistic else 0)
                for pred in Gy_rev.predecessors(node)
            )

    # Plot
    fig, ax = plt.subplots()
    for b in blocks:
        w, h = sizes[b]
        ax.add_patch(plt.Rectangle((x[b], y[b]), w, h, edgecolor='black', facecolor='lightblue'))
        ax.text(x[b] + w / 2, y[b] + h / 2, b, ha='center', va='center', fontsize=10)

    ax.set_xlim(0, max(x[b] + sizes[b][0] for b in blocks) + 1)
    ax.set_ylim(0, max(y[b] + sizes[b][1] for b in blocks) + 1)
    ax.set_aspect('equal')
    plt.title("Floorplan from Sequence Pair")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
plot_sequence_pair(model, blocks, pi_plus, pi_minus, realistic=False)
plot_sequence_pair(model, blocks, pi_plus, pi_minus, realistic=True)