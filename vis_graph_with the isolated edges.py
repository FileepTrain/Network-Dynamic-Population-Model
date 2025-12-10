import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Any, Dict, Iterable, List, Set


# ---------------------------------------------------------------------
# TYPE INFERENCE
# ---------------------------------------------------------------------
def _infer_sim_type(history, sim_type: str | None):
    """
    Infer sim_type from history if not explicitly provided.

    - 'cascade' if history[0] is a set
    - 'sirs' if history[0] is a dict with keys S, I, R (and optionally D)
    """
    if sim_type is not None:
        return sim_type

    if not history:
        raise ValueError("Empty history; cannot infer sim_type.")

    first = history[0]

    # Cascade: history is a list of sets (active/activated nodes)
    if isinstance(first, set):
        return "cascade"

    # SIRS: history is a list of dicts with S, I, R, optionally D
    if isinstance(first, dict):
        keys = set(first.keys())
        if {"S", "I", "R"}.issubset(keys):
            return "sirs"

    raise ValueError(
        "Could not infer sim_type from history. "
        "Provide sim_type='cascade' or 'sirs' explicitly."
    )


# ---------------------------------------------------------------------
# TIME SERIES PLOT
# ---------------------------------------------------------------------
def _extract_series_sirs(history: List[Dict[str, Set[Any]]]):
    S_counts = []
    I_counts = []
    R_counts = []
    D_counts = []

    for state in history:
        S_counts.append(len(state.get("S", set())))
        I_counts.append(len(state.get("I", set())))
        R_counts.append(len(state.get("R", set())))
        D_counts.append(len(state.get("D", set())))

    t = np.arange(len(history))
    return t, np.array(S_counts), np.array(I_counts), np.array(R_counts), np.array(D_counts)


def _extract_series_cascade(history: List[Set[Any]]):
    active_counts = [len(step) for step in history]
    t = np.arange(len(history))
    return t, np.array(active_counts)


def plot_simulation(
    history,
    sim_type: str | None = None,
    save_path: str | None = None,
):
    """
    Plot high-level time series for a cascade or SIRS simulation.

    Parameters
    ----------
    history : list
        For 'cascade': list of sets (activated nodes at each step).
        For 'sirs': list of dicts with keys "S", "I", "R", "D".
    sim_type : {'cascade', 'sirs'}, optional
        If None, inferred automatically from history.
    save_path : str, optional
        If provided, figure is saved to this path.
    """
    sim_type = _infer_sim_type(history, sim_type)

    plt.figure(figsize=(8, 5))

    if sim_type == "cascade":
        t, active_counts = _extract_series_cascade(history)
        plt.plot(t, active_counts, marker="o", label="Active / Adopted")
        plt.xlabel("Time step")
        plt.ylabel("# Active nodes")
        plt.title("Cascade simulation over time")

    elif sim_type == "sirs":
        t, S_counts, I_counts, R_counts, D_counts = _extract_series_sirs(history)
        plt.plot(t, S_counts, marker="o", label="S (susceptible)")
        plt.plot(t, I_counts, marker="o", label="I (infected)")
        plt.plot(t, R_counts, marker="o", label="R (recovered/immune)")

        # Only plot D if there are any deaths
        if np.any(D_counts > 0):
            plt.plot(t, D_counts, marker="o", label="D (dead)")

        plt.xlabel("Time step")
        plt.ylabel("# Nodes")
        plt.title("SIRS epidemic simulation over time")

    else:
        raise ValueError(f"Unknown sim_type '{sim_type}'. Use 'cascade' or 'sirs'.")

    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    # Non-blocking show + terminal-controlled close
    plt.show(block=False)
    print("Plot displayed. Press ENTER to close the plot window(s).")
    input()
    plt.close("all")


# ---------------------------------------------------------------------
# INTERACTIVE NETWORK VIEW
# ---------------------------------------------------------------------
def _draw_state_cascade(G: nx.Graph, active: Set[Any], pos: Dict[Any, Any]):
    colors = []
    for node in G.nodes():
        if node in active:
            colors.append("tab:red")
        else:
            colors.append("lightgray")

    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_color=colors,
        node_size=400,
        font_size=8,
    )


def _draw_state_sirs(
    G: nx.Graph,
    state: Dict[str, Set[Any]],
    pos: Dict[Any, Any],
    vaccinated=None,
):
    if vaccinated is None:
        vaccinated = set()

    S = state.get("S", set())
    I = state.get("I", set())
    R = state.get("R", set())
    D = state.get("D", set())

    # Base colors by epidemiological state
    color_map = {}
    for node in G.nodes():
        if node in D:
            color_map[node] = "black"
        elif node in I:
            color_map[node] = "tab:red"
        elif node in R:
            color_map[node] = "tab:green"
        elif node in S:
            color_map[node] = "tab:blue"
        else:
            color_map[node] = "lightgray"

    nodes_list = list(G.nodes())
    base_colors = [color_map[n] for n in nodes_list]

    # Base draw (no vaccination highlight yet)
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_color=base_colors,
        node_size=400,
        font_size=8,
    )

    # Overlay: vaccinated nodes with thick yellow border
    vaccinated = set(vaccinated)  # ensure set
    vaccinated_in_graph = [n for n in vaccinated if n in G]

    if vaccinated_in_graph:
        vacc_colors = [color_map[n] for n in vaccinated_in_graph]
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=vaccinated_in_graph,
            node_color=vacc_colors,     # keep fill color by state
            edgecolors="yellow",        # THIS is the border color
            linewidths=3.0,             # thickness of border
            node_size=450,              # slightly larger to make border visible
            label="Vaccinated",
        )


def interactive_simulation(
    G: nx.Graph,
    history,
    sim_type: str | None = None,
    layout: Dict[Any, Any] | None = None,
    vaccinated=None,
):
    """
    Step through a simulation history on the network.

    At each step, a network plot is shown. Press ENTER to go to the next step;
    type 'q' + ENTER to quit.

    Parameters
    ----------
    G : nx.Graph
        The underlying network.
    history : list
        For 'cascade': list of sets (activated nodes at each step).
        For 'sirs': list of dicts with keys S, I, R, D.
    sim_type : {'cascade', 'sirs'}, optional
        If None, inferred from history.
    layout : dict, optional
        Precomputed node positions. If None, spring_layout is used.
    """
    sim_type = _infer_sim_type(history, sim_type)

    if layout is None:
        layout = nx.spring_layout(G, seed=42)

    num_steps = len(history)
    if num_steps == 0:
        print("Empty history; nothing to display.")
        return

    for t, state in enumerate(history):
        plt.figure(figsize=(6, 5))
        plt.title(f"{sim_type.upper()} simulation – step {t}")

        if sim_type == "cascade":
            _draw_state_cascade(G, state, layout)
        elif sim_type == "sirs":
            _draw_state_sirs(
                G,
                state,
                layout,
                vaccinated=vaccinated,
            )
        else:
            raise ValueError(f"Unknown sim_type '{sim_type}'.")

        plt.axis("off")
        plt.tight_layout()
        plt.show(block=False)

        if t < num_steps - 1:
            user_input = input(
                f"Step {t}/{num_steps - 1}. Press ENTER for next, or 'q' + ENTER to quit: "
            ).strip().lower()
            plt.close()
            if user_input == "q":
                break
        else:
            print("Reached final step. Press ENTER to close.")
            input()
            plt.close()
            break
