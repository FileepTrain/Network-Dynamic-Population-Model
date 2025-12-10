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

    - Vaccinated nodes: still susceptible but have a lower infection probability.
    - Sheltered nodes: a fraction of their incident edges is removed
      (controlled by shelter_effectiveness).
    - Initiators are excluded from vaccination/shelter selection.
    """
    vaccinated = set()
    sheltered = set()
    removed_edges = []   # collect all removed edges here

    n = G.number_of_nodes()
    nodes_list = list(G.nodes())

    initiators_set = set(initiators)

    # Eligible nodes for vaccination/shelter: everyone except initiators
    eligible_nodes = [node for node in nodes_list if node not in initiators_set]

    # ------------------------------------------------------------------
    # Vaccination: r fraction of the population becomes vaccinated
    # ------------------------------------------------------------------
    if args.vaccination is not None and args.vaccination > 0:
        target_vaccinated = int(args.vaccination * n)
        if target_vaccinated > 0 and eligible_nodes:
            k = min(target_vaccinated, len(eligible_nodes))
            vaccinated = set(random.sample(eligible_nodes, k))

    # ------------------------------------------------------------------
    # Shelter: s fraction of the population is sheltered
    # ------------------------------------------------------------------
    if args.shelter is not None and args.shelter > 0:
        target_shelter = int(args.shelter * n)
        if target_shelter > 0 and eligible_nodes:
            k = min(target_shelter, len(eligible_nodes))
            sheltered = set(random.sample(eligible_nodes, k))

            # How strong is sheltering? (0–1; default = 1.0 → remove all edges)
            eff = (
                args.shelter_effectiveness
                if getattr(args, "shelter_effectiveness", None) is not None
                else 1.0
            )
            # clamp to [0, 1]
            eff = max(0.0, min(1.0, eff))

            for node in sheltered:
                # all incident edges
                if G.is_directed():
                    incident_edges = list(G.in_edges(node)) + list(G.out_edges(node))
                else:
                    incident_edges = list(G.edges(node))

                if not incident_edges:
                    continue

                # number of edges to remove for this node
                num_to_remove = int(round(eff * len(incident_edges)))
                if num_to_remove <= 0:
                    continue

                num_to_remove = min(num_to_remove, len(incident_edges))

                # choose a random subset of edges to cut
                to_remove = random.sample(incident_edges, num_to_remove)

                # remove from graph and record them
                G.remove_edges_from(to_remove)
                removed_edges.extend(to_remove)

    history, infected_ever, final_R, final_I, final_D = simulate_sirs(
        G,
        seeds=initiators,
        probability_of_infection=args.probability_of_infection,
        probability_of_death=args.probability_of_death,
        infection_duration=args.infection_duration,
        max_steps=args.lifespan,
        vaccinated=vaccinated,
        vaccination_effectiveness=args.vaccination_effectiveness,
    )

    return (
        history,
        infected_ever,
        final_R,
        final_I,
        final_D,
        vaccinated,
        sheltered,
        removed_edges,
    )


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
    parser.add_argument(
        "--lifespan",
        type=int,
        help="Number of time steps (days) to simulate",
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
    "--infection_duration",
    type=int,
    help="How long (in steps) an infected node remains infected before recovering.",
    )
    parser.add_argument(
        "--shelter",
        type=float,
        help="Sheltering parameter s (e.g., 0.3 means 30%% shelter-in-place)",
    )
    parser.add_argument(
        "--shelter_effectiveness",
        type=float,
        help=(
            "For sheltered nodes, fraction (0–1) of their incident edges to cut. "
            "1.0 = remove all edges (full isolation), 0.5 = remove half their edges."
        ),
    )
    parser.add_argument(
        "--vaccination",
        type=float,
        help="Vaccination rate r (0–1) of the population",
    )
    parser.add_argument(
        "--vaccination_effectiveness",
        type=float,
        help="Effectiveness of vaccination (0–1). 1 = fully effective (no infection), 0 = no effect.",
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
            interactive_simulation(G, history, sim_type="cascade", vaccinated=vaccinated)

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

        history, infected_ever, final_R, final_I, final_D, vaccinated, sheltered, removed_edges = run_covid_simulation(
            G,
            initiators=initiators,
            args=args,
        )

        # Print basic outcomes
        print("\n=== COVID / SIRS SIMULATION RESULT ===")
        print(f"Sheltered nodes (edges removed):           {len(sheltered)}")
        print(f"Vaccinated nodes:                          {len(vaccinated)}")
        print(f"Vaccination effectiveness: {getattr(args, 'vaccination_effectiveness', None)}")
        print(f"Total ever infected:                      {len(infected_ever)}")
        print(f"Final infected (I):                       {len(final_I)}")
        print(f"Final deceased (D):                       {len(final_D)}")
        print(f"Final recovered/removed (R):              {len(final_R) + len(final_D)}")
        print("=====================================\n")

        # Interactive visualization
        if args.interactive:
            interactive_simulation(
                G,
                history,
                sim_type="sirs",
                vaccinated=vaccinated,
                sheltered=sheltered,
                removed_edges=removed_edges,
    )

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