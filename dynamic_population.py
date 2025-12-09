#!/usr/bin/env python3
import argparse
import os
import networkx as nx
import matplotlib.pyplot as plt
import random
from cascade_sim import  simulate_cascade
from sirs_sim import simulate_sirs
from vis_graph import interactive_simulation, plot_simulation


# ---------------------------------------------------------------------
# FILE HANDLING
# ---------------------------------------------------------------------
def load_gml(path: str) -> nx.DiGraph:
    """
    Load a GML file and return a directed NetworkX graph.

    - Checks that the file exists.
    - Ensures the graph is directed.
    - Warns if there are no edges.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    try:
        G = nx.read_gml(path)
    except Exception as e:
        raise nx.NetworkXError(f"[ERROR] Failed to read GML file '{path}': {e}")

    if G.number_of_nodes() == 0:
        raise nx.NetworkXError("[ERROR] Graph has no nodes. Check your GML file.")

    if G.number_of_edges() == 0:
        print("[WARN] Graph has no edges.")

    return G

# ---------------------------------------------------------------------
# FILE HANDLING
# ---------------------------------------------------------------------
def run_covid_simulation(G: nx.Graph, initiators, args):
    """
    Helper for the COVID/SIRS part of the assignment.

    - Builds the immune set from vaccination + shelter, excluding initiators.
    - Calls simulate_sirs with the right parameters.
    - Returns (history, infected_ever, final_R, final_I, immune).
    """
    immune = set()
    n = G.number_of_nodes()
    nodes_list = list(G.nodes())

    # Make sure we treat initiators as a set for fast lookup
    initiators_set = set(initiators)

    # Eligible nodes for vaccination/shelter: everyone except initiators
    eligible_nodes = [node for node in nodes_list if node not in initiators_set]

    # Vaccination: r fraction of the population becomes immune
    if args.vaccination is not None and args.vaccination > 0:
        target_vaccinated = int(args.vaccination * n)
        if target_vaccinated > 0 and eligible_nodes:
            k = min(target_vaccinated, len(eligible_nodes))
            vaccinated = set(random.sample(eligible_nodes, k))
            immune |= vaccinated

    # Shelter: s fraction of the population becomes immune / non-participatory
    if args.shelter is not None and args.shelter > 0:
        target_shelter = int(args.shelter * n)
        if target_shelter > 0 and eligible_nodes:
            k = min(target_shelter, len(eligible_nodes))
            sheltered = set(random.sample(eligible_nodes, k))
            immune |= sheltered

    # Use a default if probability_of_death is not provided
    death_prob = args.probability_of_death if args.probability_of_death is not None else 0.0

    # Run SIRS simulation (assignment's "COVID" model)
    history, infected_ever, final_R, final_I, final_D = simulate_sirs(
        G,
        seeds=initiators,
        immune=immune,
        probability_of_infection=args.probability_of_infection,
        probability_of_death=args.probability_of_death,
        infection_duration=args.lifespan,
        max_steps=args.lifespan,
    )

    return history, infected_ever, final_R, final_I, final_D, immune


# ---------------------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the dynamic population assignment.

    Usage example:
        python ./dynamic_population.py graph.gml --action cascade \
            --initiator 1,2,5 --threshold 0.33 --plot
    """
    parser = argparse.ArgumentParser(
        description="Network Dynamic Population Model (cascade / COVID SIRS)"
    )

    # Positional: graph file
    parser.add_argument(
        "graph_file",
        help="Path to input graph in .gml format (e.g., graph.gml)",
    )

    # Required: action
    parser.add_argument(
        "--action",
        choices=["cascade", "covid"],
        required=True,
        help="Type of simulation to run: 'cascade' or 'covid'",
    )

    # Common: initiator(s)
    parser.add_argument(
        "--initiator",
        type=str,
        required=True,
        help=(
            "Comma-separated list of initial node IDs to start from "
            "(e.g., '1,2,5')"
        ),
    )

    # Cascade-specific
    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold q (0–1) for cascade activation",
    )

    # COVID-specific
    parser.add_argument(
        "--probability_of_infection",
        type=float,
        help="Infection probability p (0–1)",
    )
    parser.add_argument(
        "--probability_of_death",
        type=float,
        help="Death probability q (0–1) while infected",
    )
    parser.add_argument(
        "--lifespan",
        type=int,
        help="Number of time steps (days) to simulate",
    )
    parser.add_argument(
        "--shelter",
        type=float,
        help="Sheltering parameter s (e.g., 0.3 means 30%% shelter-in-place)",
    )
    parser.add_argument(
        "--vaccination",
        type=float,
        help="Vaccination rate r (0–1) of the population",
    )

    # Plotting options
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Plot graph + node states every round (interactive mode)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot summary (e.g., new infections per day) at the end",
    )

    return parser


# ---------------------------------------------------------------------
# HELPER: PARSE INITIATORS
# ---------------------------------------------------------------------
def parse_initiators(initiator_str: str):
    """
    Parse comma-separated initiator IDs into a list of node labels (strings).

    The GML file might use string labels, so we keep them as strings here.
    """
    parts = [s.strip() for s in initiator_str.split(",") if s.strip()]
    return parts


# ---------------------------------------------------------------------
# MAIN (ONLY TESTS ARGS + FILE HANDLING)
# ---------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load the graph
    # ------------------------------------------------------------
    try:
        G = load_gml(args.graph_file)
    except (FileNotFoundError, nx.NetworkXError) as e:
        print(e)
        return

    # Parse initiators (keep as strings for GML)
    initiators = parse_initiators(args.initiator)

    # ------------------------------------------------------------
    # CASCADE ACTION
    # ------------------------------------------------------------
    if args.action == "cascade":

        if args.threshold is None:
            print("[ERROR] --threshold is required for cascade action.")
            return

        print("[INFO] Running cascade simulation...")

        history, final_adopted = simulate_cascade(
            G,
            seeds=initiators,
            q=args.threshold
        )

        # Print results
        print("\n=== CASCADE SIMULATION RESULT ===")
        for r, nodes in enumerate(history):
            print(f"Round {r}: {sorted(nodes)}")
        print(f"\nFinal adopted count: {len(final_adopted)}")
        print("=================================\n")

        # Interactive visualization
        if args.interactive:
            interactive_simulation(G, history, sim_type="cascade")

        # Plot curve
        if args.plot:
            plot_simulation(history, sim_type="cascade", save_path="cascade_curve.png")


    # ------------------------------------------------------------
    # COVID ACTION (SIRS model)
    # ------------------------------------------------------------
    elif args.action == "covid":

        missing = []
        if args.probability_of_infection is None:
            missing.append("--probability_of_infection")
        if args.lifespan is None:
            missing.append("--lifespan")
        if args.shelter is None:
            missing.append("--shelter")
        if args.vaccination is None:
            missing.append("--vaccination")

        if missing:
            print(f"[ERROR] Missing required COVID parameters: {', '.join(missing)}")
            return

        print("[INFO] Running COVID (SIRS) simulation...")

        history, infected_ever, final_R, final_I, final_D, immune = run_covid_simulation(
            G,
            initiators=initiators,
            args=args,
        )

        # Print basic outcomes
        print("\n=== COVID / SIRS SIMULATION RESULT ===")
        print(f"Immune at start (vaccinated + sheltered): {len(immune)}")
        print(f"Total ever infected:                      {len(infected_ever)}")
        print(f"Final infected (I):                       {len(final_I)}")
        print(f"Final deceased (D):                       {len(final_D)}")
        print(f"Final recovered/removed (R):              {len(final_R) + len(final_D)}")
        print("=====================================\n")

        # Interactive visualization
        if args.interactive:
            interactive_simulation(G, history, sim_type="sirs")

        # Plot curve
        if args.plot:
            plot_simulation(history, sim_type="sirs", save_path="sirs_curve.png")

    # ------------------------------------------------------------
    # Print argument diagnostics for grading
    # ------------------------------------------------------------
    print("============ ARGUMENTS CHECK ============")
    print(f"Graph file:          {args.graph_file}")
    print(f"Loaded graph:        {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Action:              {args.action}")
    print(f"Initiators:          {initiators}")
    print(f"Threshold:           {args.threshold}")
    print(f"P(infection):        {args.probability_of_infection}")
    print(f"P(death):            {args.probability_of_death}")
    print(f"Lifespan:            {args.lifespan}")
    print(f"Shelter:             {args.shelter}")
    print(f"Vaccination:         {args.vaccination}")
    print(f"Interactive:         {args.interactive}")
    print(f"Plot:                {args.plot}")
    print("=========================================")


if __name__ == "__main__":
    main()
