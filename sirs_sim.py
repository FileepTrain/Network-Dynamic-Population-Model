import random
import networkx as nx

def simulate_sirs(
    G: nx.Graph,
    seeds,
    immune=None,
    probability_of_infection: float = 0.2,   # S → I
    probability_of_death: float = 0.1,       # I → R
    max_steps: int = 100,
):
    """
    Simulate an SIR epidemic model on graph G, recording per-step history.

    States:
      S = susceptible
      I = infected (spreading)
      R = recovered/removed (cannot be infected again)

    Args:
      G: undirected graph
      seeds: initial infected nodes (iterable)
      immune: nodes that start permanently immune → in R immediately
      probability_of_infection: P(S → I)
      probability_of_death:    P(I → R)
      max_steps: total number of time steps

    Returns:
      history: list of time steps, where each step is:
                {"S": set(...), "I": set(...), "R": set(...)}
      infected_ever: set of nodes that were *ever* infected
      final_R: nodes recovered/removed at end
      final_I: nodes still infected at end
    """

    seeds = set(seeds)
    immune = set(immune) if immune is not None else set()

    # Initialize sets
    S = set(G.nodes())
    I = set()
    R = set()

    # Immune nodes → permanently recovered
    R |= immune
    S -= immune

    # Infect initial seeds (if they are not immune)
    seeds = seeds - R
    I |= seeds
    S -= seeds

    infected_ever = set(I)

    # Save initial state
    history = [{
        "S": set(S),
        "I": set(I),
        "R": set(R),
    }]

    for _ in range(max_steps):
        if not I:
            break  # No infected left → stop

        new_I = set()
        new_R = set()

        # Infection: S → I
        for u in I:
            for v in G.neighbors(u):
                if v in S and random.random() < probability_of_infection:
                    new_I.add(v)

        # Recovery/death: I → R
        for u in I:
            if random.random() < probability_of_death:
                new_R.add(u)

        # Update states
        S -= new_I
        I |= new_I
        I -= new_R
        R |= new_R

        infected_ever |= new_I

        # Record state
        history.append({
            "S": set(S),
            "I": set(I),
            "R": set(R),
        })

    return history, infected_ever, R, I
