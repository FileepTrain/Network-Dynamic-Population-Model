import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def interactive_graph(G: nx.Graph, history, save_prefix: str = None):
    """
    Interactive infection visualization:
    Waits for the user to press ENTER to advance each round,
    then shows a final summary graph at the end.

    Parameters
    ----------
    G : networkx.Graph
    history : list of sets
        history[r] = nodes newly infected in round r.
    save_prefix : str or None
        If provided, saves each frame as "<prefix>_roundX.png".
    """
    if not history:
        print("[interactive_graph] Empty history; nothing to display.")
        return

    pos = nx.spring_layout(G, seed=42)
    infected_so_far = set()

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    print("\nInteractive Cascade Viewer")
    print("Press ENTER for next round, or type 'q' then ENTER to quit.\n")

    for r, newly_infected in enumerate(history):
        infected_so_far |= newly_infected

        ax.clear()

        # Color nodes:
        # - dark red: newly infected this round
        # - salmon: infected in previous rounds
        # - light gray: not infected yet
        node_colors = []
        for n in G.nodes():
            if n in newly_infected:
                node_colors.append("darkred")
            elif n in infected_so_far:
                node_colors.append("salmon")
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
            f"Round {r} | New: {len(newly_infected)} | Total infected: {len(infected_so_far)}"
        )
        ax.axis("off")

        # Force update to screen
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        if save_prefix is not None:
            fig.savefig(f"{save_prefix}_round{r}.png", bbox_inches="tight")

        user_input = input(
            f"Round {r} shown. Press ENTER for next, or 'q' then ENTER to quit: "
        ).strip().lower()
        if user_input == "q":
            print("Exiting interactive viewer early.")
            break

    # --- Final summary graph (no prompt) ---
    ax.clear()
    final_colors = []
    for n in G.nodes():
        if n in infected_so_far:
            final_colors.append("salmon")      # infected at some point
        else:
            final_colors.append("lightgray")   # never infected

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

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

    if save_prefix is not None:
        fig.savefig(f"{save_prefix}_final.png", bbox_inches="tight")

    plt.ioff()
    plt.show()


def plot_graph(history, save_path: str = None):
    """
    Plot the number of new infections and cumulative infections per round.

    Parameters
    ----------
    history : list[set]
        history[r] = set of nodes that became infected in round r.
    save_path : str or None
        If not None, save the figure to this path.
    """
    if not history:
        print("[plot_infection_curve] Empty history; nothing to plot.")
        return

    new_counts = [len(s) for s in history]
    cum_counts = list(np.cumsum(new_counts))
    rounds = list(range(len(history)))

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, new_counts, marker="o", label="New infections")
    plt.plot(rounds, cum_counts, marker="s", linestyle="--", label="Cumulative infections")

    plt.xlabel("Round")
    plt.ylabel("Number of nodes")
    plt.title("Infection / Cascade over time")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def interactive_sir_graph(G: nx.Graph, history, save_prefix: str = None):
    """
    Interactive SIR visualization.
    Shows S/I/R states at each time step; user advances with ENTER.

    Colors:
      - gray: S (susceptible)
      - red:  I (currently infected)
      - green: R (recovered/removed)

    Parameters
    ----------
    G : networkx.Graph
    history : list of dicts
        history[t] = {"S": set(...), "I": set(...), "R": set(...)}
    save_prefix : str or None
        If provided, saves each frame as "<prefix>_tX.png" and a final as "<prefix>_final.png".
    """
    if not history:
        print("[interactive_sir_graph] Empty history; nothing to display.")
        return

    pos = nx.spring_layout(G, seed=42)

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    print("\nInteractive SIR Viewer")
    print("Press ENTER for next step, or type 'q' then ENTER to quit.\n")

    last_state = history[-1]

    for t, state in enumerate(history):
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
            f"Step {t} | S={len(S)}  I={len(I)}  R={len(R)}"
        )
        ax.axis("off")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        if save_prefix is not None:
            fig.savefig(f"{save_prefix}_t{t}.png", bbox_inches="tight")

        # if this is not the last frame, ask user if they want to continue
        if t < len(history) - 1:
            user_input = input(
                f"Step {t} shown. Press ENTER for next, or 'q' then ENTER to quit: "
            ).strip().lower()
            if user_input == "q":
                print("Exiting interactive SIR viewer early.")
                break

    # Final summary graph
    final_S = last_state["S"]
    final_I = last_state["I"]
    final_R = last_state["R"]

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
        f"Final SIR State | S={len(final_S)}  I={len(final_I)}  R={len(final_R)}"
    )
    ax.axis("off")

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

    if save_prefix is not None:
        fig.savefig(f"{save_prefix}_final.png", bbox_inches="tight")

    plt.ioff()
    plt.show()