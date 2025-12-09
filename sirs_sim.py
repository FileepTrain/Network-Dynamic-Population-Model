

import random
import networkx as nx

def simulate_sirs(
    G: nx.Graph,
    seeds,
    immune=None,
    probability_of_infection: float = 0.2,   # S → I
    probability_of_death: float = 0.1,       # I → D
    infection_duration: int=5,            # Duration an individual remains infected if they don't die
    max_steps: int = 100,
):
    
    #Future implementations could make a R -> S (where whatever nodes not are "permenantly" immune)
    #probability_of_recovery: float = 0.1,   # I -> R
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
    D = set()

    # Immune nodes → permanently recovered
    R |= immune
    S -= immune

    # Infect initial seeds (if they are not immune)
    seeds = seeds - R
    I |= seeds
    S -= seeds

    infected_ever = set(I)

    #Track how long each node has been infected
    infection_age = {}
    for node in I:
        infection_age[node] = 0

    # Save initial state
    history = [{
        "S": set(S),
        "I": set(I),
        "R": set(R),
        "D": set(D),
    }]

    for _ in range(max_steps):
        if not I:
            break  # No infected left → stop

        new_I = set()
        new_R = set()
        new_D = set()

        # Infection: S → I
        for u in I:
            for v in G.neighbors(u):
                if v in S and random.random() < probability_of_infection:
                    new_I.add(v)


        # Death or Recovery: I -> D or I -> R
        for u in list(I):
            # First, check for death this step
            if random.random() < probability_of_death:
                new_D.add(u)
                infection_age.pop(u, None)  # no longer tracking age after death
            else:
                # Survives this step → increase infection age
                infection_age[u] = infection_age.get(u, 0) + 1
                # If infection has lasted long enough, recover
                if infection_age[u] >= infection_duration:
                    new_R.add(u)
                    infection_age.pop(u, None)

        #Apply new infections and initialize their infection age
        S -= new_I
        for node in new_I:
            I.add(node)
            infection_age[v] = 0    #New infections start with age 0

        infected_ever |= new_I


        # Update states
        S -= new_I
        I |= new_I
        I -= (new_R | new_D)
        R |= new_R
        D |= new_D

        infected_ever |= new_I

        # Record state
        history.append({
            "S": set(S),
            "I": set(I),
            "R": set(R),
            "D": set(D),
        })

    return history, infected_ever, R, I, D
