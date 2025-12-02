import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import linregress
import sys, glob

'''
a code to calculate in-situ Vp/Vs ratio using P and S differential 
times from a tight cluster of events. See Lin & Shearer (2007) for methodology:
"Estimating Local Vp/Vs Ratios within Similar Earthquake Clusters"
'''

# maximum allowed input cc shift
maxshift = 0.2
# FDTCC/hypoDD
input_format = 'FDTCC'
# CC threshold (applies to CC values in the input file from FDTCC only)
cc_th = 0.8
# list of input .cc/.ct files:
inputs = glob.glob('*cc')


def get_mean(dicp, dics):
    pp, ss = [], []
    for st in list(set(dicp.keys()).union(dics.keys())):
        if st in dicp.keys() and st in dics.keys():
            pp.append(float(dicp[st]))
            ss.append(float(dics[st]))
    
    return np.mean(pp), np.mean(ss)


def sort_things_out(inpfile, ccth):
    pp, ss, ccs = [], [], []
    i=0
    with open(inpfile, 'r') as inp:
        for row in inp:
            if "#" in row:
                i+=1
                if i > 1:
                    pave, save = get_mean(dict_p, dict_s)
                    for st in list(set(dict_p.keys()).union(dict_s.keys())):
                        if st in dict_p.keys() and st in dict_s.keys():
                            pp.append(float(dict_p[st]) - pave)
                            ss.append(float(dict_s[st]) - save)
                            ccs.append((p_cc[st]+s_cc[st])/2)

                dict_p = {}
                dict_s = {}
                p_cc = {}
                s_cc = {}
                continue
            
            if input_format == 'FDTCC':
                sta, diff, cc, phase = row.split()
            elif input_format == 'hypoDD':
                sta, tt1, tt2, cc, phase = row.split()
                diff = float(tt1) - float(tt2)
            else:
                print(f'\nPlease select a proper input format: FDTCC or hypoDD')
                sys.exit()

            if phase == 'P' and float(cc) >= ccth and abs(float(diff)) < maxshift:
                dict_p[sta] = diff
                p_cc[sta] = float(cc)
            elif phase == 'S' and float(cc) >= ccth and abs(float(diff)) < maxshift:
                dict_s[sta] = diff
                s_cc[sta] = float(cc)

    # evaluate the last pair of events too!
    pave, save = get_mean(dict_p, dict_s)
    for st in list(set(dict_p.keys()).union(dict_s.keys())):
        if st in dict_p.keys() and st in dict_s.keys():
            pp.append(float(dict_p[st]))
            ss.append(float(dict_s[st])) 
            ccs.append((p_cc[st]+s_cc[st])/2)

                      

    return np.array(pp), np.array(ss), np.array(ccs)



def huber_loss_sum(residuals, d_max):

    abs_res = np.abs(residuals)
    l2_mask = abs_res <= d_max
    l2_cost = 0.5 * residuals[l2_mask]**2
    l1_cost = d_max * abs_res[~l2_mask] - 0.5 * d_max**2
    
    return np.sum(l2_cost) + np.sum(l1_cost)


def get_robust_mean(data, d_max=None):

    if d_max is None:
        mad = np.median(np.abs(data - np.median(data)))
        d_max = 1.345 * mad if mad > 0 else 1.0

    def objective(mu):
        return huber_loss_sum(data - mu, d_max)

    result = minimize_scalar(objective)
    return result.x


def fit_robust_line_slope(x, y, d_max):
    def line_cost(slope):
        algebraic_dist = y - slope * x
        norm_factor = np.sqrt(1 + slope**2)
        intercept_shift = get_robust_mean(algebraic_dist, d_max)
        perp_residuals = (algebraic_dist - intercept_shift) / norm_factor

        return huber_loss_sum(perp_residuals, d_max)

    res = minimize_scalar(line_cost, bounds=(0.1, 10.0), method='bounded')
    return res.x


def calculate_vp_vs_ratio(dt_p, dt_s, initial_R=1.732, max_iter=10, tol=1e-9):

    R = initial_R
    mad_p = np.median(np.abs(dt_p - np.median(dt_p)))
    d_max = 1.345 * mad_p if mad_p > 0 else 0.1
    
    print(f"Starting Iteration with R={R:.4f}, d_max={d_max:.4f}")

    for i in range(max_iter):
        prev_R = R
        dt_s_scaled = dt_s / R
        mu_p = get_robust_mean(dt_p, d_max)
        mu_s = get_robust_mean(dt_s_scaled, d_max)
        
        x_centered = dt_p - mu_p
        y_centered = dt_s_scaled - mu_s
        slope = fit_robust_line_slope(x_centered, y_centered, d_max)
        R = R * slope
        
        print(f"Iter {i+1}: Slope found={slope:.4f}, Updated R={R:.4f}")
        if np.abs(R - prev_R) < tol:
            print("Converged.")
            break
            
    return R



def plot_vp_vs_fit(dt_p, dt_s, estimated_R, filename, ccs, d_max=None):

    if d_max is None:
        mad_p = np.median(np.abs(dt_p - np.median(dt_p)))
        d_max = 1.345 * mad_p if mad_p > 0 else 0.1
        
    mu_p = get_robust_mean(dt_p, d_max)
    mu_s = get_robust_mean(dt_s, d_max)
    x_line = np.linspace(min(dt_p), max(dt_p), 100)
    y_robust = estimated_R * (x_line - mu_p) + mu_s
    slope_ls, intercept_ls, _, _, _ = linregress(dt_p, dt_s)
    y_ls = slope_ls * x_line + intercept_ls

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(dt_p, dt_s, c=ccs, cmap='viridis', label='Observed Data')
    cbar = plt.colorbar(sc)
    cbar.set_label('CC values')
    plt.plot(x_line, y_robust, color='red', linewidth=2.5, 
             label=f'Robust Fit (R={estimated_R:.3f})')
    plt.plot(x_line, y_ls, color='blue', linestyle='--', alpha=0.7,
             label=f'Standard Least Sq (Slope={slope_ls:.3f})')

    plt.xlabel("Differential P-time ($dt_p$)")
    plt.ylabel("Differential S-time ($dt_s$)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=450)
    plt.show()




for ccfile in inputs:
    diffs_p, diffs_s, ave_ccs = sort_things_out(ccfile, cc_th)
    if len(diffs_p) < 4: continue

    outname = ccfile.split('/')[-1].replace('.cc', '') + ".png"
    estimated_R = calculate_vp_vs_ratio(diffs_p, diffs_s)
    plot_vp_vs_fit(diffs_p, diffs_s, estimated_R, outname, ave_ccs)

