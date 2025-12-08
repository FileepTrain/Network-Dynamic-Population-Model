#!/usr/bin/env python3
import argparse
import os
import networkx as nx
import matplotlib.pyplot as plt
from cascade_sim import  simulate_cascade
from vis_graph import interactive_graph, plot_graph


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

    # Load the graph
    try:
        G = load_gml(args.graph_file)
    except (FileNotFoundError, nx.NetworkXError) as e:
        print(e)
        return

    # Parse initiators
    initiators = parse_initiators(args.initiator)

    if args.action == "cascade":
        if args.threshold is None:
            print("[WARN] --threshold is missing for cascade action.")
        else:
            print("[INFO] Running CASCADE test...")

            # --- Run cascade simulation ---
            history, final_adopted = simulate_cascade(
                G,
                seeds=initiators,
                q=args.threshold
            )

            print("\n=== CASCADE SIMULATION RESULT ===")
            for r, nodes in enumerate(history):
                print(f"Round {r}: {sorted(nodes)}")
            print(f"\nFinal adopted set ({len(final_adopted)} nodes): {sorted(final_adopted)}")
            print("=================================\n")


    if args.action == "covid":
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
            print(f"[WARN] Missing COVID parameters: {', '.join(missing)}")
        else:
            print("[INFO] Running COVID test mode (no simulation yet).")
            
    history, final_adopted = simulate_cascade(
        G,
        seeds=initiators,
        q=args.threshold
    )

    # For interactive graph animation:
    if args.interactive:
        interactive_graph(G, history)

    # For summary plot at the end:
    if args.plot:
        plot_graph(history, save_path="infection_curve.png")

    print(f"Graph file:          {args.graph_file}")
    print(f"Loaded graph:        {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Action:              {args.action}")
    print(f"Initiators (raw):    {args.initiator}")
    print(f"Initiators (parsed): {initiators}")
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
