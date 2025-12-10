import random
import networkx as nx

def simulate_sirs(
    G: nx.Graph,
    seeds,
    probability_of_infection: float = 0.2,   # S → I
    probability_of_death: float = 0.1,       # I → D
    infection_duration: int = 5,             # steps infected if they don't die
    max_steps: int = 100,
    vaccinated=None,
    vaccination_effectiveness: float = 0.5,  # in [0, 1], 1 = fully effective, 0 = no effect
):
    """
    Simulate an SIRS epidemic model on graph G, recording per-step history.

    States:
      S = susceptible
      I = infected (spreading)
      R = recovered (no longer infected)
      D = dead

    Args:
      G: graph (typically undirected)
      seeds: initial infected nodes (iterable)
      probability_of_infection: P(S → I) per contact per step
      probability_of_death:    P(I → D) per step
      infection_duration: number of steps an individual remains infected
                          if they do NOT die in the meantime
      max_steps: total number of simulation steps
      vaccinated: iterable of vaccinated nodes (may be None)
      vaccination_effectiveness: scales infection probability:
                                eff=0   -> behaves like unvaccinated
                                eff=1   -> cannot be infected (p -> 0)

    Returns:
      history: list of time steps, where each element is:
               {"S": set(...), "I": set(...), "R": set(...), "D": set(...)}
      infected_ever: set of nodes that were ever infected
      final_R: nodes recovered at end
      final_I: nodes still infected at end
      final_D: nodes dead at end
    """

    # --- Handle None values by using defaults ---
    if probability_of_infection is None:
        probability_of_infection = 0.2

    if probability_of_death is None:
        probability_of_death = 0.1

    if infection_duration is None:
        infection_duration = 5

    if max_steps is None:
        max_steps = 100

    if vaccination_effectiveness is None:
        vaccination_effectiveness = 0.5

    # --- Vaccination set and clamp effectiveness ---
    seeds = set(seeds)
    vaccinated = set(vaccinated) if vaccinated is not None else set()
    vaccination_effectiveness = max(0.0, min(1.0, vaccination_effectiveness))

    # --- Initialize S, I, R, D ---
    S = set(G.nodes())
    I = set()
    R = set()
    D = set()

    # Infect initial seeds
    seeds = seeds - R
    I |= seeds
    S -= seeds

    infected_ever = set(I)

    # Track how long each node has been infected
    infection_age = {node: 0 for node in I}

    # Initial state
    history = [{
        "S": set(S),
        "I": set(I),
        "R": set(R),
        "D": set(D),
    }]

    # --- Main simulation loop ---
    for _ in range(max_steps):
        if not I:
            break  # No infected left → stop

        new_I = set()
        new_R = set()
        new_D = set()

        # Infection: S → I
        for u in I:
            for v in G.neighbors(u):
                if v not in S:
                    continue

                # Base infection probability
                p = probability_of_infection

                # Vaccinated nodes get lower p
                if v in vaccinated:
                    p = probability_of_infection * (1.0 - vaccination_effectiveness)

                if p > 0.0 and random.random() < p:
                    new_I.add(v)

        # Death or Recovery: I → D or I → R
        for u in list(I):
            # Death
            if random.random() < probability_of_death:
                new_D.add(u)
                infection_age.pop(u, None)
            else:
                # Survives this step → accumulate infection age
                infection_age[u] = infection_age.get(u, 0) + 1
                if infection_age[u] >= infection_duration:
                    new_R.add(u)
                    infection_age.pop(u, None)

        # Initialize infection age for newly infected
        for node in new_I:
            infection_age[node] = 0

        # Update compartments
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
