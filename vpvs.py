import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


'''

a code to calculate in situ Vp/Vs ratio using P and S differential 
times from a tight cluster of events.

Input file: dt.cc from hypoDD program
-input file format (see example dt.cc file):
# event_ID1     event_ID2
station     differential_time       CC    phase

'''


ccfile = 'dt.cc'        # input file
cc_th = 0.8             # cc threshold (applies to CC values in the input file)


def get_mean(dicp, dics):
    pp, ss = [], []
    for st in list(set(dicp.keys()).union(dics.keys())):
        if st in dicp.keys() and st in dics.keys():
            pp.append(float(dicp[st]))
            ss.append(float(dics[st]))
    
    return np.mean(pp), np.mean(ss)


def sort_things_out(inpfile, ccth):
    pp, ss = [], []
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

                dict_p = {}
                dict_s = {}                
                continue
        
            sta, diff, cc, phase = row.split()
            if phase == 'P' and float(cc) >= ccth:
                dict_p[sta] = diff
            elif phase == 'S' and float(cc) >= ccth:
                dict_s[sta] = diff

    # evaluate the last pair of events too!
    pave, save = get_mean(dict_p, dict_s)
    for st in list(set(dict_p.keys()).union(dict_s.keys())):
        if st in dict_p.keys() and st in dict_s.keys():
            pp.append(float(dict_p[st]))
            ss.append(float(dict_s[st]))            
                      

    return np.array(pp), np.array(ss)



def huber_weight(residuals, delta=1.0):
    # Compute Huber weights based on residuals
    return 1 / (1 + (residuals / delta) ** 2)


def irls_regression(X, y, max_iter=10, tol=1e-6, delta=1.0):
    # Perform Iteratively Reweighted Least Squares (IRLS) regression

    X = np.c_[np.ones(X.shape[0]), X]  
    weights = np.ones(len(y)) 

    for _ in range(max_iter):
        W = np.diag(weights)  
        beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y  

        residuals = y - X @ beta
        new_weights = huber_weight(residuals, delta=delta)

        if np.max(np.abs(new_weights - weights)) < tol:
            break
        
        weights = new_weights

    return beta 


# get pairs of P and S differential travel times
diffs_p, diffs_s = sort_things_out(ccfile, cc_th)

# fit the regression line
diffs_p = diffs_p.reshape(-1, 1)
beta = irls_regression(diffs_p, diffs_s)
intercept, slope = beta[0], beta[1]
equation = fr"$V_p/V_s = {slope:.2f}x + {intercept:.2f}$"
x_range = np.linspace(min(diffs_p)-0.1, max(diffs_p)+0.1, 100)
y_pred = intercept + slope * x_range

# plot the results
plt.scatter(diffs_p, diffs_s, color='blue', label=equation)
plt.plot(x_range, y_pred, color='red', linewidth=2)
plt.xlabel(r"$\Delta t_p$ (s)")
plt.ylabel(r"$\Delta t_s$ (s)")
plt.legend()
plt.savefig('vp_vs_ratio.jpg', dpi=600)
plt.show()