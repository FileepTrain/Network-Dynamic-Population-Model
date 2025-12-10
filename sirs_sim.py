import random
import networkx as nx

def simulate_sirs(
    G: nx.Graph,
    seeds,
    probability_of_infection: float = 0.2,   # S → I
    probability_of_death: float = 0.1,       # I → D
    infection_duration: int = 5,             # time infected before recovering
    max_steps: int = 100,
    vaccinated=None,
    vaccination_effectiveness: float = 0.5,
    resusceptibility: float = 0.1,           # R → S per step
):
    """
    Simulate an SIRS epidemic model on graph G.

    States:
      S = susceptible
      I = infected (spreading)
      R = recovered (temporarily immune)
      D = dead (removed)
    """

    # --- Handle None defaults ---
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
    if resusceptibility is None:
        resusceptibility = 0.1

    # Clamp all probabilities to [0,1]
    probability_of_infection = max(0.0, min(1.0, probability_of_infection))
    probability_of_death = max(0.0, min(1.0, probability_of_death))
    vaccination_effectiveness = max(0.0, min(1.0, vaccination_effectiveness))
    resusceptibility = max(0.0, min(1.0, resusceptibility))

    seeds = set(seeds)
    vaccinated = set(vaccinated) if vaccinated is not None else set()

    # Compartment sets
    S = set(G.nodes())
    I = set()
    R = set()
    D = set()

    # Infect initial seeds
    I |= seeds
    S -= seeds

    infected_ever = set(I)

    # Track infection ages
    infection_age = {node: 0 for node in I}

    # History list
    history = [{
        "S": set(S),
        "I": set(I),
        "R": set(R),
        "D": set(D),
    }]

    for _ in range(max_steps):
        if not I and not R:
            break

        new_I = set()
        new_R = set()
        new_D = set()
        new_S_from_R = set()

        # ------------------------
        # R → S (resusceptibility)
        # ------------------------
        if resusceptibility > 0:
            for u in list(R):
                if random.random() < resusceptibility:
                    new_S_from_R.add(u)

        R -= new_S_from_R
        S |= new_S_from_R

        # ------------------------
        # S → I (infection)
        # ------------------------
        for u in I:
            for v in G.neighbors(u):
                if v not in S:
                    continue

                p = probability_of_infection
                if v in vaccinated:
                    p = probability_of_infection * (1.0 - vaccination_effectiveness)

                if p > 0 and random.random() < p:
                    new_I.add(v)

        # ------------------------
        # I → D or I → R
        # ------------------------
        for u in list(I):
            if random.random() < probability_of_death:
                new_D.add(u)
                infection_age.pop(u, None)
            else:
                infection_age[u] = infection_age.get(u, 0) + 1
                if infection_age[u] >= infection_duration:
                    new_R.add(u)
                    infection_age.pop(u, None)

        # Start infection age for new cases
        for node in new_I:
            infection_age[node] = 0

        # Update states
        S -= new_I
        I |= new_I
        I -= (new_R | new_D)
        R |= new_R
        D |= new_D

        infected_ever |= new_I

        history.append({
            "S": set(S),
            "I": set(I),
            "R": set(R),
            "D": set(D),
        })

    return history, infected_ever, R, I, D
