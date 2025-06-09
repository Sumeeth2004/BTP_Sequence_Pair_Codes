from z3 import *
import matplotlib.pyplot as plt
import networkx as nx
import random

blocks = ['A', 'B', 'C', 'D']
trial_input = [([('A','B'),'C'], 'v'), (['A', 'D', 'B'], 'vo')]

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

print("\nEnumerating all sequence pairs for given constraints of: " +str(trial_input)+"\n")
all_solutions = enumerate_all_seq_pairs(blocks, trial_input)