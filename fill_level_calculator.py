import io
import base64
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # non-GUI backend for Flask
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ============================================================
# 0. Load cost data (once at import)
# ============================================================
df = pd.read_csv("tanks_cost_data_2003_michigan.csv")


def run_tank_network_analysis(tanks, adjacency, environment=None):
    """
    Main entry point called from Flask.

    Parameters
    ----------
    tanks : list of dict
        Each element has keys:
        - id
        - diameter_m
        - height_m
        - specific_gravity
        - fill_level_m

    adjacency : 2D list (n x n)
        0/1 adjacency matrix from connectivity table.

    environment : dict
        Expected keys (from Table 3 in frontend):
        - "Flood Depth (m)"
        - "Water Velocity (m/s)"
        - "Pipe Capacity (CMS)"    (m^3/s)
        - "Time Window (hr)"       (hours)
        (plus others like wind speed, etc., which we just pass through)
    """

    if environment is None:
        environment = {}

    if not tanks:
        raise ValueError("No tank data received from frontend.")

    # --------------------------------------------------------
    # 1. Build tank arrays from frontend inputs
    # --------------------------------------------------------
    n = len(tanks)

    tank_dia = np.array([t["diameter_m"] for t in tanks], dtype=float)
    tank_ht  = np.array([t["height_m"] for t in tanks], dtype=float)
    rho_avg  = np.array([t["specific_gravity"] for t in tanks], dtype=float)
    fill_level = np.array([t["fill_level_m"] for t in tanks], dtype=float)

    if len(tank_dia) != n or len(tank_ht) != n or len(rho_avg) != n or len(fill_level) != n:
        raise ValueError("Inconsistent tank array lengths from frontend.")

    # --------------------------------------------------------
    # 2. Adjacency and graph from frontend
    # --------------------------------------------------------
    tank_adj = np.array(adjacency, dtype=float)
    if tank_adj.shape != (n, n):
        raise ValueError(f"Adjacency matrix must be {n}x{n}, got {tank_adj.shape}")

    # Force symmetric 0/1 with diagonal = 1
    tank_adj = (tank_adj > 0).astype(int)
    tank_adj = np.maximum(tank_adj, tank_adj.T)
    np.fill_diagonal(tank_adj, 1)

    G = nx.from_numpy_array(tank_adj)

    clusters_raw = list(nx.connected_components(G))
    cluster_labels = np.zeros(n, dtype=int)
    for i_c, cluster in enumerate(clusters_raw):
        for node in cluster:
            cluster_labels[node] = i_c
    unique_clusters = np.unique(cluster_labels)

    # Graph layout for later plotting
    pos = nx.spring_layout(G, seed=1, k=0.9)

    # --------------------------------------------------------
    # 3. Pipe & flood data from environment (Table 3)
    # --------------------------------------------------------
    # Flood
    flood_depth = float(environment.get("Flood Depth (m)", 2.0) or 2.0)
    water_velo  = float(environment.get("Water Velocity (m/s)", 0.5) or 0.5)
    est_flood_data = [flood_depth, water_velo]

    # Pipe capacity and time window
    # Frontend gives capacity in m^3/s (CMS), convert to m^3/hr for this model
    pipe_capacity_cms = float(environment.get("Pipe Capacity (CMS)", 20.0) or (20.0))
    pipe_capacity = pipe_capacity_cms # m^3/hr
    time_window = float(environment.get("Time Window (hr)", 12.0) or 12.0)

    vol_cap_edge = pipe_capacity * time_window

    spill_unit_cost = 4000.0
    friction_coeff = 0.3

    # --------------------------------------------------------
    # 4. Build cost interpolation and tank-related cost
    # --------------------------------------------------------
    capacity_m3 = df["Capacity_m3"].values
    cost_usd = df["Cost_USD"].values
    interp_func = interp1d(capacity_m3, cost_usd, kind="linear", fill_value="extrapolate")

    tank_volume = np.pi * tank_dia**2 * tank_ht / 4.0
    tank_cost = 3.0 * interp_func(tank_volume)   # scaling factor 3 as in your code

    # Cost weight per tank (volume-related damage + tank replacement)
    volume = np.pi * tank_ht * tank_dia**2 / 4.0
    C = spill_unit_cost * volume + tank_cost

    # --------------------------------------------------------
    # 5. Logistic model for Pf and cost
    # --------------------------------------------------------
    def precompute_logit_ab():
        x1 = tank_dia
        x2 = tank_ht
        x3 = rho_avg
        x6 = est_flood_data[0]  # depth
        x7 = 0.0
        x8 = 0.0
        x9 = est_flood_data[1]  # velocity
        x10 = 0.0
        x11 = friction_coeff

        # Constant terms in logit (independent of fill level x4)
        a = (
            1.42e1
            - x1*6.63e-1 - x2*2.30e-1 - x3*2.76 + x6*1.82e1 - x7*1.37e1 - x8*6.29e-1
            + x9*2.04 + x10*6.12e-2 - x11*5.32e1
            - x1*x7*2.68e-1 - x1*x8*2.01e-1 - x1*x10*1.26e-3 + x1*x11*1.26
            + x6*x7*1.06 + x6*x8*6.66e-1 + x7*x8*5.63
            + x1*x1*7.88e-2 + x11*x11*4.01e1
            - x1*x7*x8*1.05e-1
            + x1*x1*x7*9.10e-3 + x1*x1*x8*3.46e-3 - x1*x1*x11*2.36e-2
            - x1*x1*x1*1.86e-3 + x1*x1*x1*x1*1.27e-5
        )

        # Coefficients multiplying fill level x4 → b
        b = (
            5.51
            - x1*2.24e-1
            - x3*2.05e1
            - x11*6.18
            + x1*x11*1.39e-1
            + x1*x1*1.89e-3
        )

        return np.asarray(a, dtype=float), np.asarray(b, dtype=float)

    a_logit, b_logit = precompute_logit_ab()

    def cost_func(x):
        z = a_logit + b_logit * x
        pf = 1.0 / (1.0 + np.exp(-z))
        return np.sum(C * pf)

    def cost_grad(x):
        z = a_logit + b_logit * x
        sig = 1.0 / (1.0 + np.exp(-z))
        dpf = sig * (1.0 - sig)
        return C * dpf * b_logit

    def cost_hess(x, v=None):
        z = a_logit + b_logit * x
        sig = 1.0 / (1.0 + np.exp(-z))
        dpf = sig * (1.0 - sig)
        d2pf = dpf * (1.0 - 2.0 * sig)
        diag = C * d2pf * (b_logit**2)
        return np.diag(diag)

    # --------------------------------------------------------
    # 6. Volume-preserving constraints per cluster
    # --------------------------------------------------------
    def volume_constraints_cluster():
        Aeq = []
        beq = []
        for uc in unique_clusters:
            mask = (cluster_labels == uc)
            row = np.zeros(n, dtype=float)
            row[mask] = np.pi * tank_dia[mask]**2 / 4.0  # area
            Aeq.append(row)
            beq.append(np.sum(row * fill_level))  # initial volume per cluster
        return np.array(Aeq), np.array(beq)

    Aeq, beq = volume_constraints_cluster()

    eq_cons_unconstrained = []
    for i in range(len(beq)):
        Ai = Aeq[i].copy()
        bi = beq[i]
        eq_cons_unconstrained.append({
            'type': 'eq',
            'fun': (lambda x, A=Ai, b=bi: np.dot(A, x) - b),
            'jac': (lambda x, A=Ai: A)
        })

    bounds_x = [(0.0, 0.9 * ht) for ht in tank_ht]

    # --------------------------------------------------------
    # 7. Solve unconstrained (no pipe/time limits, volume preserved)
    # --------------------------------------------------------
    fill_init = fill_level.copy()

    res_unconstrained = minimize(
        fun=cost_func,
        x0=fill_level,
        method='trust-constr',
        jac=cost_grad,
        hess=cost_hess,
        bounds=bounds_x,
        constraints=eq_cons_unconstrained,
        options={'maxiter': 2000, 'gtol': 1e-10, 'xtol': 1e-12}
    )

    fill_level_opt = res_unconstrained.x
    cost_orig = cost_func(fill_level)
    cost_opt_unconstrained = cost_func(fill_level_opt)

    # --------------------------------------------------------
    # 8. Build edge list, incidence matrix, and integrated problem
    # --------------------------------------------------------
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if tank_adj[i, j] > 0:
                edges.append((i, j))

    m = len(edges)

    B = np.zeros((n, m))
    for e_idx, (i, j) in enumerate(edges):
        B[i, e_idx] = -1.0
        B[j, e_idx] = +1.0

    area = np.pi * tank_dia**2 / 4.0

    def cost_func_full(z):
        x = z[:n]
        return cost_func(x)

    def cost_grad_full(z):
        x = z[:n]
        gx = cost_grad(x)
        gf = np.zeros(m)
        return np.concatenate([gx, gf])

    def cost_hess_full(z, v=None):
        x = z[:n]
        Hx = cost_hess(x)
        H_full = np.zeros((n + m, n + m))
        H_full[:n, :n] = Hx
        return H_full

    def node_balance(z):
        x = z[:n]
        f = z[n:]
        dV = area * (x - fill_init)
        return B @ f - dV

    def node_balance_jac(z):
        J = np.zeros((n, n + m))
        J[:, :n] = -np.diag(area)
        J[:, n:] = B
        return J

    eq_cons_integrated = [{
        'type': 'eq',
        'fun':  node_balance,
        'jac':  node_balance_jac
    }]

    x_bounds = [(0.0, 0.9 * ht) for ht in tank_ht]
    f_bounds = [(-vol_cap_edge, vol_cap_edge)] * m
    bounds_full = x_bounds + f_bounds

    z0 = np.concatenate([fill_level, np.zeros(m)])

    res_integrated = minimize(
        fun=cost_func_full,
        x0=z0,
        method='trust-constr',
        jac=cost_grad_full,
        hess=cost_hess_full,
        bounds=bounds_full,
        constraints=eq_cons_integrated,
        options={'maxiter': 2000, 'gtol': 1e-10, 'xtol': 1e-12}
    )

    z_opt = res_integrated.x
    fill_level_time_opt = z_opt[:n]
    flows_opt = z_opt[n:]
    cost_time_feasible = cost_func(fill_level_time_opt)

    # --------------------------------------------------------
    # 9. Build plots and encode as base64
    # --------------------------------------------------------
    # Figure 1: Fill levels bar plot
    tank_idx = np.arange(n)
    bar_w = 0.35

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.bar(tank_idx - bar_w/2, fill_init, bar_w, label="Initial")
    ax1.bar(tank_idx + bar_w/2, fill_level_time_opt, bar_w, label="Time-feasible optimized")
    ax1.set_xlabel("Tank Index")
    ax1.set_ylabel("Fill Level (m)")
    ax1.set_title("Fill Levels: Initial vs Time-Feasible Optimized")
    ax1.legend()
    fig1.tight_layout()

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=150)
    plt.close(fig1)
    buf1.seek(0)
    fill_bar_b64 = base64.b64encode(buf1.read()).decode('ascii')

    # Figure 2: Network flow plot
    delta_fill = fill_level_time_opt - fill_init
    norm = mcolors.TwoSlopeNorm(
        vcenter=0.0,
        vmin=float(delta_fill.min()),
        vmax=float(delta_fill.max())
    )
    cmap = cm.get_cmap("RdYlGn")
    node_colors = cmap(norm(delta_fill))

    abs_flows = np.abs(flows_opt)
    if abs_flows.size > 0 and abs_flows.max() > 0:
        max_width = 6.0
        edge_widths = max_width * abs_flows / abs_flows.max()
    else:
        edge_widths = np.ones_like(abs_flows)

    fig2, ax2 = plt.subplots(figsize=(7, 5))

    # Draw edges with widths
    for e_idx, (i, j) in enumerate(edges):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(i, j)],
            width=edge_widths[e_idx],
            edge_color="tab:blue",
            alpha=0.7,
            ax=ax2,
        )

    nx.draw_networkx_nodes(
        G, pos,
        node_size=300,
        node_color=node_colors,
        edgecolors="k",
        ax=ax2,
    )
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax2)
    ax2.set_title("Network Flow ")
    # ax2.axis("off")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Fill Level Change (m)")

    fig2.tight_layout()

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=150)
    plt.close(fig2)
    buf2.seek(0)
    network_flow_b64 = base64.b64encode(buf2.read()).decode('ascii')

    # --------------------------------------------------------
    # 10. Build result dict for frontend
    # --------------------------------------------------------
    result = {
    "tanks": tanks,
    "environment": environment,
    "adjacency": tank_adj.tolist(),
    "cluster_labels": cluster_labels.tolist(),
    "num_clusters": int(len(unique_clusters)),

    "original_cost": float(cost_orig),
    "unconstrained_cost": float(cost_opt_unconstrained),
    "time_feasible_cost": float(cost_time_feasible),

    "fill_initial": fill_init.tolist(),
    "fill_unconstrained_opt": fill_level_opt.tolist(),    
    "fill_time_feasible_opt": fill_level_time_opt.tolist(),

    "fill_bar_plot_png_base64": fill_bar_b64,
    "network_flow_plot_png_base64": network_flow_b64,
}
    return result
