import random
import networkx as nx
from typing import Iterable, Set, Dict, List, Tuple, Any


def simulate_sirs(
    G: nx.Graph,
    seeds: Iterable[Any],
    immune: Iterable[Any] | None = None,
    probability_of_infection: float = 0.2,   # S → I
    probability_of_death: float = 0.1,       # I → D
    infection_duration: int = 5,             # Duration an individual remains infected if they do not die
    max_steps: int = 100,
    vaccinated: Iterable[Any] | None = None,
    vaccination_effectiveness: float = 0.0,  # in [0, 1]; 1 = fully effective, 0 = no effect
) -> Tuple[
    List[Dict[str, Set[Any]]],
    Set[Any],
    Set[Any],
    Set[Any],
    Set[Any],
]:
    """
    Simple SIRS-like epidemic simulation on a static graph.

    States:
        S: susceptible
        I: infected
        R: recovered / immune
        D: dead

    Parameters
    ----------
    G : nx.Graph
        Contact network.
    seeds : iterable
        Initial infected nodes.
    immune : iterable, optional
        Nodes that start immune (in R) and never leave that state.
    probability_of_infection : float
        Base probability that an S node becomes infected from an infected neighbor.
    probability_of_death : float
        Probability that an infected node dies at each time step.
    infection_duration : int
        Minimum number of steps a node remains infected (if it does not die earlier).
    max_steps : int
        Maximum number of simulation steps.
    vaccinated : iterable, optional
        Nodes that are vaccinated. They are still susceptible but with a reduced
        effective infection probability.
    vaccination_effectiveness : float in [0, 1]
        Effectiveness of vaccination. Interpreted as:
            effective_p_infection = probability_of_infection * (1 - vaccination_effectiveness)
        for vaccinated nodes.

    Returns
    -------
    history : list of dict
        Each element is {"S": set, "I": set, "R": set, "D": set} for a time step.
    infected_ever : set
        Nodes that were ever infected at any point.
    R : set
        Nodes recovered/immune at the end.
    I : set
        Nodes infected at the end.
    D : set
        Nodes dead at the end.
    """

    # Normalize inputs
    seeds = set(seeds)
    immune = set(immune) if immune is not None else set()
    vaccinated = set(vaccinated) if vaccinated is not None else set()

    # Clamp vaccination effectiveness
    vaccination_effectiveness = max(0.0, min(1.0, vaccination_effectiveness))

    nodes = set(G.nodes())

    # Initial state
    I: Set[Any] = set(seeds)
    # immune nodes start in R and are never allowed in S or I
    R: Set[Any] = set(immune)
    D: Set[Any] = set()
    S: Set[Any] = nodes - I - R - D

    # Track how long each node has been infected
    infection_age: Dict[Any, int] = {u: 0 for u in I}

    infected_ever: Set[Any] = set(I)

    history: List[Dict[str, Set[Any]]] = []

    # Record initial state (t = 0)
    history.append({
        "S": set(S),
        "I": set(I),
        "R": set(R),
        "D": set(D),
    })

    for _ in range(max_steps):
        if not I:
            # No active infections; epidemic ended
            break

        new_I: Set[Any] = set()
        new_R: Set[Any] = set()
        new_D: Set[Any] = set()

        # Infection: S → I
        for u in I:
            if u not in G:  # node might have been removed in some external manipulation
                continue
            for v in G.neighbors(u):
                # Only susceptible and non-immune nodes can be infected
                if v not in S:
                    continue
                if v in immune:
                    continue

                # Base infection probability
                p = probability_of_infection

                # Vaccinated nodes: lower effective probability
                if v in vaccinated:
                    # effectiveness scales the infection probability down
                    p = probability_of_infection * (1.0 - vaccination_effectiveness)

                if p > 0.0 and random.random() < p:
                    new_I.add(v)

        # Disease progression for currently infected nodes
        for u in list(I):
            # Increase infection age
            infection_age[u] = infection_age.get(u, 0) + 1

            # First, check death
            if random.random() < probability_of_death:
                new_D.add(u)
                continue

            # If not dead and has been infected long enough, recover
            if infection_age[u] >= infection_duration:
                new_R.add(u)

        # Initialize age for new infections
        for node in new_I:
            infection_age[node] = 0

        # State transitions
        S -= new_I
        I |= new_I
        I -= (new_R | new_D)
        R |= new_R
        D |= new_D

        infected_ever |= new_I

        # Remove infection_age entries for nodes that left I
        for node in new_R | new_D:
            if node in infection_age:
                del infection_age[node]

        # Record state
        history.append({
            "S": set(S),
            "I": set(I),
            "R": set(R),
            "D": set(D),
        })

        # No changes → steady state
        if not new_I and not new_R and not new_D:
            break

    return history, infected_ever, R, I, D
