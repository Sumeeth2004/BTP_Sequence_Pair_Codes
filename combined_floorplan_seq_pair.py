"""
Combined Floorplan and Sequence Pair Enumeration
"""

import math
import random
import matplotlib.pyplot as plt
from z3 import *
import itertools
class Module:
    def __init__(self, name, area, aspect_ratios):
        self._name = name
        self._area = area
        self._aspect_ratios = aspect_ratios
        self._width = None
        self._height = None
        self._x = None
        self._y = None

    def __repr__(self):
        return f"Module({self._name}, area={self._area}, aspect_ratios={self._aspect_ratios})"

def costEval(F):
    # Compute the bounding rectangle area of the floorplan F
    max_x = 0
    max_y = 0
    for m in F:
        if isinstance(m, Module):
            if m._x is None or m._y is None:
                continue
            max_x = max(max_x, m._x + m._width)
            max_y = max(max_y, m._y + m._height)
    return max_x * max_y

def findCoord(F, pi_plus=None, pi_minus=None):
    """
    Assign coordinates to modules in F using the longest path-based compaction method based on sequence pair indices.
    If pi_plus and pi_minus are not provided, fall back to simple left-to-right placement.
    """
    modules = [m for m in F if isinstance(m, Module)]
    if pi_plus is None or pi_minus is None:
        # Fallback: simple left-to-right placement as before
        x_cursor = 0
        y_cursor = 0
        max_height_in_row = 0
        coords = {}
        for item in F:
            if isinstance(item, Module):
                ar = item._aspect_ratios[0]
                w = math.sqrt(item._area * ar)
                h = item._area / w
                item._width = w
                item._height = h
                item._x = x_cursor
                item._y = y_cursor
                coords[item._name] = (item._x, item._y, item._width, item._height)
                x_cursor += w
                if h > max_height_in_row:
                    max_height_in_row = h
            elif item == 'V':
                y_cursor += max_height_in_row
                x_cursor = 0
                max_height_in_row = 0
        return coords

    # Use the longest path algorithm based on sequence pair
    # Assign width and height for each module (using first aspect ratio)
    for m in modules:
        ar = m._aspect_ratios[0]
        w = math.sqrt(m._area * ar)
        h = m._area / w
        m._width = w
        m._height = h

    # Build fast lookup for indices
    pi_plus_idx = {b: i for i, b in enumerate(pi_plus)}
    pi_minus_idx = {b: i for i, b in enumerate(pi_minus)}
    module_names = [m._name for m in modules]
    # Build horizontal and vertical constraint graphs (corrected logic for longest path)
    h_graph = {b: [] for b in module_names}
    v_graph = {b: [] for b in module_names}
    # Corrected vertical and horizontal graph formation for the longest path approach
    for i in range(len(module_names)):
        for j in range(len(module_names)):
            if i == j:
                continue
            a = module_names[i]
            b = module_names[j]
            # In pi_plus and pi_minus: if b before a in both, b left of a
            if pi_plus_idx[b] < pi_plus_idx[a] and pi_minus_idx[b] < pi_minus_idx[a]:
                h_graph[b].append(a)
            # In pi_plus and reverse in pi_minus: if b before a in pi_plus and after in pi_minus, b below a
            if pi_plus_idx[b] < pi_plus_idx[a] and pi_minus_idx[b] > pi_minus_idx[a]:
                v_graph[a].append(b)  # b below a
    # Longest path: x for horizontal, y for vertical
    x = {b: 0 for b in module_names}
    y = {b: 0 for b in module_names}
    # Topological order
    def topo_sort(graph):
        visited = set()
        order = []
        def dfs(u):
            if u in visited:
                return
            visited.add(u)
            for v in graph[u]:
                dfs(v)
            order.append(u)
        for u in graph:
            dfs(u)
        return order[::-1]
    # Compute x
    for u in topo_sort(h_graph):
        for v in h_graph[u]:
            x[v] = max(x[v], x[u] + next(m._width for m in modules if m._name == u))
    # Compute y
    for u in topo_sort(v_graph):
        for v in v_graph[u]:
            y[v] = max(y[v], y[u] + next(m._height for m in modules if m._name == u))
    coords = {}
    for m in modules:
        m._x = x[m._name]
        m._y = y[m._name]
        coords[m._name] = (m._x, m._y, m._width, m._height)
    return coords

def plot(coords):
    fig, ax = plt.subplots()
    for name, (x, y, w, h) in coords.items():
        rect = plt.Rectangle((x, y), w, h, fill=None, edgecolor='r')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center')
    ax.set_xlim(0, max(x + w for x, y, w, h in coords.values()) * 1.1)
    ax.set_ylim(0, max(y + h for x, y, w, h in coords.values()) * 1.1)
    ax.set_aspect('equal')
    plt.show()

def enumerate_all_seq_pairs(blocks, input):
    n = len(blocks)

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

    solutions = []
    while s.check() == sat:
        model = s.model()
        pi_plus_list = sorted(blocks, key=lambda b: model[pi_plus[b]].as_long())
        pi_minus_list = sorted(blocks, key=lambda b: model[pi_minus[b]].as_long())
        print("π₊:", pi_plus_list)
        print("π₋:", pi_minus_list,"\n")
        solutions.append((model, pi_plus_list, pi_minus_list))

        # Improved blocking clause to prevent duplicate (pi_plus, pi_minus) pairs
        same_pi_plus = And(*[pi_plus[b] == model[pi_plus[b]].as_long() for b in blocks])
        same_pi_minus = And(*[pi_minus[b] == model[pi_minus[b]].as_long() for b in blocks])
        s.add(Not(And(same_pi_plus, same_pi_minus)))
    if not solutions:
        print("No valid sequence pairs found under constraints.")
    return solutions

def main():
    # Sample input blocks and constraints (from previous test in seq_pair_enumeration.py)
    blocks = ['A', 'W', 'X', 'Y', 'Z']

    trial_input = [
        ([('X', 'Y'), 'Z'], 'h'),
        (['Z', 'A'], 'vo')
    ]

    module_defs = {
        'X': Module('X', 25, [1]),
        'Y': Module('Y', 30, [1]),
        'Z': Module('Z', 10, [1]),
        'W': Module('W', 20, [1]),
        'A': Module('A', 15, [1])
    }
    modules = [module_defs[b] for b in blocks]

    # Enumerate all valid sequence pairs
    print("Enumerating all sequence pairs under constraints...\n")
    all_seq_pairs = enumerate_all_seq_pairs(blocks, trial_input)

    if not all_seq_pairs:
        print("No valid sequence pairs found.")
        return

    best_area = math.inf
    best_floorplan = None
    best_coords = None
    best_seq_pair = None
    best_aspect_combo = None
    best_coords_list = []
    best_seq_pair_list = []
    best_aspect_combo_list = []

    for idx, (model, pi_plus_list, pi_minus_list) in enumerate(all_seq_pairs):
        print(f"Processing sequence pair {idx+1}:")
        print("  π₊:", pi_plus_list)
        print("  π₋:", pi_minus_list)

        # Build initial floorplan F according to the sequence pair (π₊, π₋)
        # The sequence pair representation is not directly a Polish expression.
        # For simplicity, use π₊ as the module order and insert 'V' operators between them.
        base_modules = [module_defs[b] for b in pi_plus_list]
        # Generate all combinations of aspect ratios for modules in this order
        aspect_ratio_lists = [m._aspect_ratios for m in base_modules]
        for aspect_combo in itertools.product(*aspect_ratio_lists):
            # Assign aspect ratios to modules
            for m, ar in zip(base_modules, aspect_combo):
                m._width = math.sqrt(m._area * ar)
                m._height = m._area / m._width
            # Build floorplan F with 'V' operators between modules
            F = []
            for i, m in enumerate(base_modules):
                F.append(m)
                if i != len(base_modules) - 1:
                    F.append('V')
            # Compute coordinates and area
            coords = findCoord(F, pi_plus=pi_plus_list, pi_minus=pi_minus_list)
            area = costEval(F)
            if area < best_area:
                best_area = area
                best_floorplan = F
                best_coords = coords
                best_seq_pair = (pi_plus_list, pi_minus_list)
                best_aspect_combo = aspect_combo
                best_coords_list = [coords]  # reset list with new best
                best_seq_pair_list = [ (pi_plus_list, pi_minus_list) ]
                best_aspect_combo_list = [ aspect_combo ]
            elif area == best_area:
                best_coords_list.append(coords)
                best_seq_pair_list.append( (pi_plus_list, pi_minus_list) )
                best_aspect_combo_list.append( aspect_combo )

    print("\nBest floorplan(s) found:")
    print("  π₊:", best_seq_pair[0])
    print("  π₋:", best_seq_pair[1])
    print("  Aspect ratios:", best_aspect_combo)
    print(f"  Area: {best_area}")

    # Print info for each best floorplan before plotting
    n_best = len(best_coords_list)
    if n_best == 0:
        print("No best floorplans to plot.")
        return

    # Select up to 8 random floorplans if more than 8 exist
    if n_best > 8:
        indices = random.sample(range(n_best), 8)
        best_coords_list_selected = [best_coords_list[i] for i in indices]
        best_seq_pair_list_selected = [best_seq_pair_list[i] for i in indices]
        best_aspect_combo_list_selected = [best_aspect_combo_list[i] for i in indices]
        n_plot = 8
    else:
        best_coords_list_selected = best_coords_list
        best_seq_pair_list_selected = best_seq_pair_list
        best_aspect_combo_list_selected = best_aspect_combo_list
        n_plot = n_best

    for i in range(n_plot):
        pi_plus_list, pi_minus_list = best_seq_pair_list_selected[i]
        aspect_combo = best_aspect_combo_list_selected[i]
        coords = best_coords_list_selected[i]
        area = costEval(best_floorplan)  # area is same for all best, but recalc if needed
        print(f"\nBest floorplan #{i+1}:")
        print("  π₊:", pi_plus_list)
        print("  π₋:", pi_minus_list)
        print("  Aspect ratios:", aspect_combo)
        print(f"  Area: {area}")

    # Arrange plots in 2 rows and 4 columns for better readability
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    axes_flat = axes.flatten()
    padding = 0.08  # 8% padding around modules
    for idx in range(nrows * ncols):
        if idx < n_plot:
            ax = axes_flat[idx]
            coords = best_coords_list_selected[idx]
            # Compute bounds
            max_x = max(x + w for x, y, w, h in coords.values())
            max_y = max(y + h for x, y, w, h in coords.values())
            min_x = min(x for x, y, w, h in coords.values())
            min_y = min(y for x, y, w, h in coords.values())
            dx = max_x - min_x
            dy = max_y - min_y
            pad_x = dx * padding
            pad_y = dy * padding
            # Draw rectangles and labels
            for name, (x, y, w, h) in coords.items():
                rect = plt.Rectangle((x, y), w, h, fill=None, edgecolor='r', linewidth=2)
                ax.add_patch(rect)
                ax.text(
                    x + w / 2,
                    y + h / 2,
                    name,
                    ha='center',
                    va='center',
                    fontsize=18,
                    fontweight='bold',
                    color='navy'
                )
            # Titles and axis labels
            area = max_x * max_y
            ax.set_title(f"Floorplan #{idx+1}\nArea: {area:.2f}", fontsize=16)
            ax.set_xlabel("X", fontsize=14)
            ax.set_ylabel("Y", fontsize=14)
            # Set limits with padding
            ax.set_xlim(min_x - pad_x, max_x + pad_x)
            ax.set_ylim(min_y - pad_y, max_y + pad_y)
            ax.set_aspect('equal')
            # Remove grid, minimize ticks
            ax.grid(False)
            ax.tick_params(axis='both', which='both', length=0, labelsize=10)
            # Remove or limit ticks/numbers for clarity
            ax.set_xticks([])
            ax.set_yticks([])
            # Optional: draw light border for axes
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('#888')
        else:
            # Hide unused axes
            axes_flat[idx].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()