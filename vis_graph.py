import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# ---------------------------------------------------------------------
# TYPE INFERENCE
# ---------------------------------------------------------------------
def _infer_sim_type(history, sim_type):
    """
    Infer sim_type from history if not explicitly provided.

    - 'cascade' if history[0] is a set
    - 'sirs' if history[0] is a dict with keys S, I, R
    """
    if sim_type is not None:
        return sim_type.lower()

    if not history:
        raise ValueError("Cannot infer sim_type from empty history.")

    first = history[0]
    if isinstance(first, set):
        return "cascade"
    if isinstance(first, dict) and all(k in first for k in ("S", "I", "R")):
        return "sirs"

    raise ValueError(
        "Could not infer sim_type from history. "
        "Pass sim_type='cascade' or sim_type='sirs' explicitly."
    )


# ---------------------------------------------------------------------
# DRAW HELPERS (ONE FRAME)
# ---------------------------------------------------------------------
def _draw_cascade_frame(G, pos, ax, newly_infected, infected_so_far, round_idx):
    """Draw a single cascade frame."""
    ax.clear()

    node_colors = []
    for n in G.nodes():
        if n in newly_infected:
            node_colors.append("darkred")      # new this round
        elif n in infected_so_far:
            node_colors.append("salmon")       # previously infected
        else:
            node_colors.append("lightgray")    # never infected

    nx.draw_networkx_edges(
        G, pos,
        edge_color="lightgray",
        width=1.5,
        alpha=0.7,
        ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=450,
        ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(
        f"Round {round_idx} | New: {len(newly_infected)} | Total infected: {len(infected_so_far)}"
    )
    ax.axis("off")


def _draw_cascade_final(G, pos, ax, infected_so_far):
    """Draw final cascade summary frame."""
    ax.clear()
    final_colors = [
        "salmon" if n in infected_so_far else "lightgray"
        for n in G.nodes()
    ]

    nx.draw_networkx_edges(
        G, pos,
        edge_color="lightgray",
        width=1.5,
        alpha=0.7,
        ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_color=final_colors,
        node_size=450,
        ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(f"Final infection state | Total infected: {len(infected_so_far)}")
    ax.axis("off")


def _draw_sirs_frame(G, pos, ax, state, step_idx):
    """Draw a single SIRS frame."""
    S = state["S"]
    I = state["I"]
    R = state["R"]

    ax.clear()

    node_colors = []
    for n in G.nodes():
        if n in I:
            node_colors.append("red")
        elif n in R:
            node_colors.append("green")
        else:
            node_colors.append("lightgray")

    nx.draw_networkx_edges(
        G, pos,
        edge_color="lightgray",
        width=1.5,
        alpha=0.7,
        ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=450,
        ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(
        f"Step {step_idx} | S={len(S)}  I={len(I)}  R={len(R)} (SIRS)"
    )
    ax.axis("off")


def _draw_sirs_final(G, pos, ax, final_state):
    """Draw final SIRS summary frame."""
    final_S = final_state["S"]
    final_I = final_state["I"]
    final_R = final_state["R"]

    ax.clear()
    final_colors = []
    for n in G.nodes():
        if n in final_I:
            final_colors.append("red")
        elif n in final_R:
            final_colors.append("green")
        else:
            final_colors.append("lightgray")

    nx.draw_networkx_edges(
        G, pos,
        edge_color="lightgray",
        width=1.5,
        alpha=0.7,
        ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_color=final_colors,
        node_size=450,
        ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(
        f"Final SIRS State | S={len(final_S)}  I={len(final_I)}  R={len(final_R)}"
    )
    ax.axis("off")


# ---------------------------------------------------------------------
# INTERACTIVE RUNNERS (PER SIM TYPE)
# ---------------------------------------------------------------------
def _run_interactive_cascade(G, history, pos, fig, ax, save_prefix):
    print("\nInteractive Cascade Viewer")
    print("Press ENTER for next round, or type 'q' then ENTER to quit.\n")

    infected_so_far = set()

    for r, newly_infected in enumerate(history):
        infected_so_far |= newly_infected

        # draw frame
        _draw_cascade_frame(G, pos, ax, newly_infected, infected_so_far, r)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        if save_prefix is not None:
            fig.savefig(f"{save_prefix}_round{r}.png", bbox_inches="tight")

        # terminal output
        print(f"[Round {r}]")
        print(f"  Newly infected ({len(newly_infected)}): {sorted(newly_infected)}")
        print(f"  Total infected so far: {len(infected_so_far)}\n")

        user_input = input(
            f"Round {r} shown. Press ENTER for next, or 'q' then ENTER to quit: "
        ).strip().lower()
        if user_input == "q":
            print("Exiting interactive cascade viewer early.")
            break

    # final frame
    _draw_cascade_final(G, pos, ax, infected_so_far)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

    if save_prefix is not None:
        fig.savefig(f"{save_prefix}_final.png", bbox_inches="tight")


def _run_interactive_sirs(G, history, pos, fig, ax, save_prefix):
    print("\nInteractive SIRS Viewer")
    print("Press ENTER for next step, or type 'q' then ENTER to quit.\n")

    last_state = history[-1]

    for t, state in enumerate(history):
        S = state["S"]
        I = state["I"]
        R = state["R"]

        # draw frame
        _draw_sirs_frame(G, pos, ax, state, t)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        if save_prefix is not None:
            fig.savefig(f"{save_prefix}_t{t}.png", bbox_inches="tight")

        # terminal output
        if t == 0:
            prev_I = set()
            prev_R = set()
        else:
            prev_I = history[t - 1]["I"]
            prev_R = history[t - 1]["R"]

        new_infections = I - prev_I - prev_R
        new_deaths = R - prev_R  # newly moved into R this step

        print(f"[Step {t}]")
        print(f"  S = {len(S)}, I = {len(I)}, R = {len(R)}")
        print(f"  New infections this step: {len(new_infections)} -> {sorted(new_infections)}")
        print(f"  New deaths/recoveries this step: {len(new_deaths)} -> {sorted(new_deaths)}\n")

        if t < len(history) - 1:
            user_input = input(
                f"Step {t} shown. Press ENTER for next, or 'q' then ENTER to quit: "
            ).strip().lower()
            if user_input == "q":
                print("Exiting interactive SIRS viewer early.")
                break

    _draw_sirs_final(G, pos, ax, last_state)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

    if save_prefix is not None:
        fig.savefig(f"{save_prefix}_final.png", bbox_inches="tight")


# ---------------------------------------------------------------------
# PUBLIC: INTERACTIVE SIMULATION
# ---------------------------------------------------------------------
def interactive_simulation(G: nx.Graph, history, sim_type: str = None, save_prefix: str = None):
    """
    Unified interactive visualization.

    For 'cascade':
        history[r] = set of newly infected nodes at round r.

    For 'sirs':
        history[t] = {'S': set(...), 'I': set(...), 'R': set(...)}.
    """
    if not history:
        print("[interactive_simulation] Empty history; nothing to display.")
        return

    sim_type = _infer_sim_type(history, sim_type)
    pos = nx.spring_layout(G, seed=42)

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    if sim_type == "cascade":
        _run_interactive_cascade(G, history, pos, fig, ax, save_prefix)
    elif sim_type == "sirs":
        _run_interactive_sirs(G, history, pos, fig, ax, save_prefix)
    else:
        raise ValueError(f"Unknown sim_type '{sim_type}'. Use 'cascade' or 'sirs'.")

    plt.ioff()

    # Allow user to close from terminal instead of GUI
    print("Interactive visualization complete. Press ENTER to close the plot window(s).")
    input()
    plt.close("all")


# ---------------------------------------------------------------------
# PLOTTING HELPERS (TIME SERIES)
# ---------------------------------------------------------------------
def _compute_cascade_series(history):
    """Return (rounds, new_counts, cum_counts) for cascade."""
    new_counts = [len(s) for s in history]
    cum_counts = list(np.cumsum(new_counts))
    rounds = list(range(len(history)))
    return rounds, new_counts, cum_counts


def _compute_sirs_series(history):
    """
    Return (rounds, new_infections, cum_ever, deaths_cum) for SIRS.

    - new_infections[t]: newly infected at step t
    - cum_ever[t]: number of nodes ever infected by step t
    - deaths_cum[t]: cumulative deaths/removals = |R_t|
    """
    T = len(history)
    rounds = list(range(T))

    new_infections = []
    cum_ever = []
    deaths_cum = []

    ever = set()

    for t in range(T):
        I_t = history[t]["I"]
        R_t = history[t]["R"]

        if t == 0:
            new_t = set(I_t)  # seeds at time 0
        else:
            prev_I = history[t - 1]["I"]
            prev_R = history[t - 1]["R"]
            new_t = I_t - prev_I - prev_R

        ever |= I_t | R_t
        new_infections.append(len(new_t))
        cum_ever.append(len(ever))
        deaths_cum.append(len(R_t))   # cumulative deaths/removals

    return rounds, new_infections, cum_ever, deaths_cum


# ---------------------------------------------------------------------
# PUBLIC: TIME-SERIES PLOT
# ---------------------------------------------------------------------
def plot_simulation(history, sim_type: str = None, save_path: str = None):
    """
    Unified plotting function.

    For 'cascade':
        history[r] = set of newly infected nodes at round r.

    For 'sirs':
        history[t] = {'S': set(...), 'I': set(...), 'R': set(...)}.
    """
    if not history:
        print("[plot_simulation] Empty history; nothing to plot.")
        return

    sim_type = _infer_sim_type(history, sim_type)

    if sim_type == "cascade":
        rounds, new_counts, cum_counts = _compute_cascade_series(history)

        plt.figure(figsize=(8, 5))
        plt.plot(rounds, new_counts, marker="o", label="New infections")
        plt.plot(rounds, cum_counts, marker="s", linestyle="--", label="Cumulative infections")
        plt.title("Cascade Infection / Adoption over Time")
        plt.xlabel("Round")
        plt.ylabel("Number of nodes")

    elif sim_type == "sirs":
        rounds, new_infections, cum_ever, deaths_cum = _compute_sirs_series(history)

        plt.figure(figsize=(8, 5))
        plt.plot(rounds, new_infections, marker="o", label="New infections per step")
        plt.plot(rounds, cum_ever, marker="s", linestyle="--", label="Cumulative ever infected")
        plt.plot(rounds, deaths_cum, marker="^", linestyle="-.", label="Cumulative deaths (R)")
        plt.title("SIRS Infection Dynamics")
        plt.xlabel("Time step")
        plt.ylabel("Number of nodes")

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
