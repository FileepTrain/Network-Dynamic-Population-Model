def simulate_cascade(G, seeds, q=0.2, max_steps=50):
    """
    Multi-seed threshold cascade that records which nodes adopt in each round.

    Returns:
        history: list of sets
            history[0] = set of initial seeds
            history[1] = nodes newly adopted in round 1
            history[2] = nodes newly adopted in round 2
            ...
        final_adopted: set
            All nodes that ever adopted (union of all rounds)
    """
    adopted = set(seeds)

    # history[0] = initial seeds
    history = [set(adopted)]

    for _ in range(max_steps):
        new = set()

        for node in G.nodes():
            if node in adopted:
                continue
            neigh = list(G.neighbors(node))
            if not neigh:
                continue
            active = sum(1 for v in neigh if v in adopted)
            if active / len(neigh) >= q:
                new.add(node)

        if not new:
            break

        adopted |= new
        history.append(new)   # store only the nodes that adopted *this* round

    # final set of all adopted nodes
    final_adopted = adopted
    return history, final_adopted