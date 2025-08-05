import numpy as np
import math
from sympy import symbols, solve, Lt

def round_up_to_near(x):
    return round(math.ceil(x * 2000 ) / 2000, 4)

def calculate_shell_thickness_per_course_us(D, H_total, G, CA, Sd, St, course_height=8.0):
    segment_heights, thicknesses = [], []
    current_height = 0.0
    while current_height < H_total:
        h_segment = min(course_height, H_total - current_height)
        segment_heights.append(h_segment)
        H_segment = H_total - current_height
        td = (2.6 * D * (H_segment - 1) * G) / Sd + 0.0625  # 0.0625 = 1/16 of an inch which is corrosion allowance 
        tt = (2.6 * D * (H_segment - 1)) / St
        t_required = max(td, tt)
        t_required = max(t_required, 0.25)
        thicknesses.append(round(t_required, 4))
        current_height += h_segment
    return segment_heights, thicknesses

def get_Cf(h, D):
    hd_values = [1, 7, 25]
    cf_values = [0.7, 0.8, 0.9]
    hd_ratio = h / D
    if hd_ratio <= hd_values[0]:
        return cf_values[0]
    elif hd_ratio >= hd_values[-1]:
        return cf_values[-1]
    for i in range(len(hd_values) - 1):
        if hd_values[i] <= hd_ratio <= hd_values[i + 1]:
            x0, x1 = hd_values[i], hd_values[i + 1]
            y0, y1 = cf_values[i], cf_values[i + 1]
            return round(y0 + (hd_ratio - x0) * (y1 - y0) / (x1 - x0), 3)

def calculate_b(H, D):
    hd = H / D
    if hd <= 0.25:
        return 0.2 * D
    elif hd == 0.5:
        return 0.5 * D
    elif hd >= 1.0:
        return 0.1 * H + 0.6 * D
    else:
        x0, y0 = 0.25, 0.2 * D
        x1, y1 = 0.5, 0.5 * D
        return round(y0 + (hd - x0) * (y1 - y0) / (x1 - x0), 3)

def compute_MinFill_cone(V, D, h_f, G):
    H = 48
    f = 5
    # G = 1.0
    CA = 0.0625
    Sd = 23200
    St = 24900
    rho_steel = 0.284
    alpha = 9.5
    Z_g = 900
    Gust = 0.85
    K_d = K_zt = K_e = 1
    GC_pi = 0.18
    mu = 0.4
    h_p = symbols("h_p")

    segment_heights, thicknesses = calculate_shell_thickness_per_course_us(D, H, G, CA, Sd, St)
    sum_ht = np.sum(np.array(thicknesses) * (12 * np.array(segment_heights)))
    shell_wt = np.pi * (D*12) * sum_ht * rho_steel
    bottom_wt = np.pi * (D*12)**2 * 0.25 * 0.25 * rho_steel
    roof_wt = 1.014 * D**2.6229
    wt_tank = shell_wt + bottom_wt + roof_wt
    moment_wt_tank = wt_tank * D / 2

    buoyancy = np.pi * D**2 * 0.25 * h_f * 62.4
    moment_bucy = buoyancy * D / 2

    K_z = 2.01 * ((H/2)/Z_g)**(2/alpha)
    K_h = 2.01 * ((H+f/3)/Z_g)**(2/alpha)
    q_z = 0.00256 * K_z * K_zt *K_d * K_e * V**2
    q_h = 0.00256 * K_h * K_zt *K_d * K_e * V**2

    windF_s = q_z * Gust * get_Cf(H, D) * D * H 
    windM_s = windF_s * H/2

    c_p1, c_p2 = -0.8, -0.5
    P1 = q_h * ((Gust * c_p1 ) - GC_pi)
    b = calculate_b(H, D)
    A1 = (D/2)**2 * np.arccos((D/2 - b) / (D/2)) - ((D/2) - b) * np.sqrt((D/2)**2 - ((D/2) - b)**2)
    F1 = -P1 * A1
    b_percent = (b - (D/2))/(D/2)
    x1 = (D/2) - (0.53*b_percent - 0.425) * (D/2)
    M_zone1 = F1 * x1

    P2 = q_h * ((Gust * c_p2 ) - GC_pi)
    A2 = np.pi * D**2 * 0.25 - A1
    F2 = -P2 * A2
    x2 = (D/2) - (0.53*b_percent +  0.425) * (D/2)
    M_zone2 = F2 * x2 

    F_roof = F1 + F2
    M_roof = M_zone1 + M_zone2

    Weight_p = h_p * (np.pi * D**2 * 0.25) * (62.4 * G)
    Moment_p = Weight_p * D/2

    sumFy = buoyancy + F_roof - wt_tank - Weight_p
    F_friction = - mu * sumFy

    Fx_inequality = Lt(windF_s - F_friction, 0)
    Fy_inequality = Lt(buoyancy + F_roof - wt_tank - Weight_p, 0)
    M_inequality = Lt(moment_bucy + M_roof - moment_wt_tank - Moment_p, 0)

    h_p_sliding = solve(Fx_inequality, h_p).args[0].lhs
    h_p_floatation = solve(Fy_inequality, h_p).args[0].lhs
    h_p_overturning = solve(M_inequality, h_p).args[0].lhs

    return {
        "Min. fill required to prevent sliding failure": float(h_p_sliding),
        "Min. fill required to prevent floatation failure": float(h_p_floatation),
        "Min. fill required to prevent overturning failure": float(h_p_overturning)
    }
# print(compute_rfl_fixed(120, 100, 10, 1.0))