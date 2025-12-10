#!/usr/bin/env python3
import argparse
import os
import random
from typing import Any, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from cascade_sim import simulate_cascade
from sirs_sim import simulate_sirs
from vis_graph import interactive_simulation, plot_simulation


# ---------------------------------------------------------------------
# FILE HANDLING
# ---------------------------------------------------------------------
def load_gml(path: str) -> nx.DiGraph:
    """
    Load a graph from a .gml file.

    Parameters
    ----------
    path : str
        Path to the .gml file.

    Returns
    -------
    G : nx.Graph or nx.DiGraph
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")

    G = nx.read_gml(path)
    return G


# ---------------------------------------------------------------------
# INITIATORS PARSING
# ---------------------------------------------------------------------
def parse_initiators(initiator_str: str | None, G: nx.Graph) -> List[Any]:
    """
    Parse a comma-separated list of initiator node IDs.

    If None or empty, pick a single random initiator.
    """
    if initiator_str is None or initiator_str.strip() == "":
        # Default: choose a single random node
        node = random.choice(list(G.nodes()))
        return [node]

    tokens = [tok.strip() for tok in initiator_str.split(",") if tok.strip()]
    # Node labels in GML are often strings; keep as-is
    return tokens


# ---------------------------------------------------------------------
# COVID / SIRS SIMULATION HELPER
# ---------------------------------------------------------------------
def run_covid_simulation(
    G: nx.Graph,
    initiators: Iterable[Any],
    args: argparse.Namespace,
) -> Tuple[
    list,
    Set[Any],
    Set[Any],
    Set[Any],
    Set[Any],
    Set[Any],
    Set[Any],
    Set[Any],
]:
    """
    Run the SIRS-based COVID-like simulation with shelter and vaccination.

    Semantics:
    - Vaccinated nodes: still susceptible but with a reduced infection probability.
    - Sheltered nodes: all incident edges are removed (cannot infect / be infected via edges).
    - Initiators are excluded from vaccination / shelter selection.

    Returns
    -------
    history, infected_ever, final_R, final_I, final_D, immune, vaccinated, sheltered
    """
    immune: Set[Any] = set()      # reserved for explicit immunity, if needed later
    vaccinated: Set[Any] = set()
    sheltered: Set[Any] = set()

    n = G.number_of_nodes()
    nodes_list = list(G.nodes())
    initiators_set = set(initiators)

    # Eligible nodes for vaccination/shelter: everyone except initiators
    eligible_nodes = [node for node in nodes_list if node not in initiators_set]

    # Vaccination: r fraction of the population becomes vaccinated (but not immune)
    if getattr(args, "vaccination", None) is not None and args.vaccination > 0:
        target_vaccinated = int(args.vaccination * n)
        if target_vaccinated > 0 and eligible_nodes:
            k = min(target_vaccinated, len(eligible_nodes))
            vaccinated = set(random.sample(eligible_nodes, k))

    # Shelter: s fraction of the population has all edges removed
    if getattr(args, "shelter", None) is not None and args.shelter > 0:
        target_shelter = int(args.shelter * n)
        if target_shelter > 0 and eligible_nodes:
            k = min(target_shelter, len(eligible_nodes))
            sheltered = set(random.sample(eligible_nodes, k))

            # Remove all edges incident to sheltered nodes
            for node in sheltered:
                if G.is_directed():
                    # remove both in- and out-edges
                    in_edges = list(G.in_edges(node))
                    out_edges = list(G.out_edges(node))
                    G.remove_edges_from(in_edges + out_edges)
                else:
                    G.remove_edges_from(list(G.edges(node)))

    # Default for probability_of_death if not provided
    death_prob = (
        args.probability_of_death
        if getattr(args, "probability_of_death", None) is not None
        else 0.0
    )

    # Vaccination effectiveness (0–1, default 0 → no effect)
    vacc_eff = (
        args.vaccination_effectiveness
        if getattr(args, "vaccination_effectiveness", None) is not None
        else 0.0
    )

    # Run SIRS simulation
    history, infected_ever, final_R, final_I, final_D = simulate_sirs(
        G,
        seeds=initiators,
        immune=immune,
        probability_of_infection=args.probability_of_infection,
        probability_of_death=death_prob,
        infection_duration=args.lifespan,
        max_steps=args.lifespan,
        vaccinated=vaccinated,
        vaccination_effectiveness=vacc_eff,
    )

    return (
        history,
        infected_ever,
        final_R,
        final_I,
        final_D,
        immune,
        vaccinated,
        sheltered,
    )


# ---------------------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser.

    Example:
        python dynamic_population.py graph.gml --action cascade \
            --initiator 1,2,5 --threshold 0.33 --plot
    """
    parser = argparse.ArgumentParser(
        description="Network Dynamic Population Model (cascade / COVID SIRS)"
    )

    # Positional: graph file
    parser.add_argument(
        "graph_file",
        help="Path to input graph in .gml format (e.g., graph.gml).",
    )

    # Action: cascade vs COVID/SIRS
    parser.add_argument(
        "--action",
        choices=["cascade", "covid"],
        required=True,
        help="Which simulation to run: 'cascade' (threshold model) or 'covid' (SIRS).",
    )

    # Initiators
    parser.add_argument(
        "--initiator",
        type=str,
        default=None,
        help="Comma-separated list of initiator node IDs. "
             "If omitted, a random node is chosen.",
    )

    # Cascade parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.33,
        help="Threshold for cascade model (fraction of active neighbors).",
    )

    # SIRS / COVID parameters
    parser.add_argument(
        "--probability_of_infection",
        type=float,
        default=0.2,
        help="Base probability of infection per I–S contact (S -> I).",
    )

    parser.add_argument(
        "--probability_of_death",
        type=float,
        default=0.1,
        help="Probability that an infected node dies at each time step (I -> D).",
    )

    parser.add_argument(
        "--lifespan",
        type=int,
        default=20,
        help="Number of time steps to simulate (and infection duration for SIRS).",
    )

    parser.add_argument(
        "--shelter",
        type=float,
        default=0.0,
        help="Fraction of nodes to shelter (all edges removed) in COVID/SIRS mode.",
    )

    parser.add_argument(
        "--vaccination",
        type=float,
        default=0.0,
        help="Fraction of nodes to vaccinate in COVID/SIRS mode.",
    )

    parser.add_argument(
        "--vaccination_effectiveness",
        type=float,
        default=0.0,
        help="Effectiveness of vaccination in [0, 1]. "
             "1 = fully effective (no infection), 0 = no effect.",
    )

    # Visualization options
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show an interactive step-by-step visualization of the simulation.",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the time series of the simulation.",
    )

    return parser


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load graph
    G = load_gml(args.graph_file)

    # Parse initiators
    initiators = parse_initiators(args.initiator, G)

    # Diagnostic printout of configuration
    print("=========================================")
    print("Simulation configuration")
    print("-----------------------------------------")
    print(f"Graph file:          {args.graph_file}")
    print(f"Action:              {args.action}")
    print(f"Initiators:          {initiators}")
    print(f"Threshold:           {args.threshold}")
    print(f"P(infection):        {args.probability_of_infection}")
    print(f"P(death):            {args.probability_of_death}")
    print(f"Lifespan:            {args.lifespan}")
    print(f"Shelter:             {args.shelter}")
    print(f"Vaccination:         {args.vaccination}")
    print(f"Vaccination eff.:    {args.vaccination_effectiveness}")
    print(f"Interactive:         {args.interactive}")
    print(f"Plot:                {args.plot}")
    print("=========================================\n")

    # Run chosen action
    if args.action == "cascade":
        # Assumed signature for simulate_cascade:
        #   history, activated_ever = simulate_cascade(G, seeds, threshold, max_steps)
        history, activated_ever = simulate_cascade(
            G,
            seeds=initiators,
            threshold=args.threshold,
            max_steps=args.lifespan,
        )

        print("=== CASCADE SIMULATION RESULT ===")
        print(f"Total activated ever: {len(activated_ever)}")

        if args.interactive:
            interactive_simulation(G, history, sim_type="cascade")

        if args.plot:
            plot_simulation(history, sim_type="cascade")

    elif args.action == "covid":
        (
            history,
            infected_ever,
            final_R,
            final_I,
            final_D,
            immune,
            vaccinated,
            sheltered,
        ) = run_covid_simulation(G, initiators=initiators, args=args)

        # Print basic outcomes
        print("\n=== COVID / SIRS SIMULATION RESULT ===")
        print(f"Sheltered nodes (edges removed):           {len(sheltered)}")
        print(f"Vaccinated nodes:                          {len(vaccinated)}")
        print(f"Immune at start (explicit immune set):     {len(immune)}")
        print(f"Infected at least once:                    {len(infected_ever)}")
        print(f"Final # R (recovered/immune):              {len(final_R)}")
        print(f"Final # I (infected):                      {len(final_I)}")
        print(f"Final # D (dead):                          {len(final_D)}")

        if args.interactive:
            interactive_simulation(G, history, sim_type="sirs")

        if args.plot:
            plot_simulation(history, sim_type="sirs")

    else:
        raise ValueError(f"Unknown action '{args.action}'.")


if __name__ == "__main__":
    main()
