from matplotlib import pyplot as plt
from icecream import ic
import numpy as np
import os
def identify_best_per_arm(y, yerr, find_min=True):
    if find_min:
        current_best = float("inf")
    else:
        current_best = float("-inf")
    bests = []
    bests_err = []
    for i,v in enumerate(y):
        if find_min:
            if v < current_best:
                current_best = v
                current_best_err = yerr[i]
        else:
            if v > current_best:
                current_best = v
                current_best_err = yerr[i]
        bests.append(current_best)
        bests_err.append(current_best_err)
    return np.vstack(bests), np.vstack(bests_err)
def create_convergence_plot(y, yerr,algo, folder, exp_id, replication, is_min=True):
    plt.rcParams.update({"font.size": 14})

    iters = list(range(len(y)))
    ys , yerr = identify_best_per_arm(y,yerr, find_min=is_min)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # plot the optimal value
    ax.errorbar(
        iters,
        np.mean(ys, axis=1),
        yerr=np.mean(yerr, axis=1),

        label=f"Optimium at Arm",
        color="blue",
        linewidth=1.5,
    )
    ax.set(
        xlabel=f"Arm number",
        ylabel="Best observed value",
        title=f"Convergence plot of {exp_id} with algorithm {algo}",
    )
    ax.legend(loc="best")
    
    p = os.path.join(folder, f'convergence_plot_{replication}.png')
    # parent_directory = os.path.join(os.getcwd(), os.pardir)
    # p = os.path.join(parent_directory,'plots', f'{exp_id}_{replication}_{algo}.png')
    plt.savefig(p, dpi=500)


def visualize_best_values(values, current_value, m):
    """
    Visualisiert die besten m Werte und den aktuellen Wert.
    
    :param values: Array mit den besten Werten aus vergangenen Experimenten.
    :param current_value: Wert des aktuellen Experiments.
    :param m: Anzahl der besten Werte, die angezeigt werden sollen.

    Use:
    values = np.random.rand(1000)  # 1000 zufällige Werte zwischen 0 und 1
    current_value = 0.99
    m = 50
    vis = visualize_best_values(values, current_value, m)

    """
    
    # Sortiere die Werte und wähle die besten m Werte aus
    best_m_values = np.sort(values)[-m:]
    
    # Erstelle eine Liste von y-Werten (alle gleich 1)
    y = [0] * m
    
    # Plotte die besten m Werte als graue Punkte
    plt.scatter(best_m_values, y, color='lightgray')
    
    # Plotte den aktuellen Wert als roten Punkt
    plt.scatter(current_value, 0, color='red', s=100)  # s=100 macht den Punkt größer
    
    # Entferne die Achsenbeschriftungen
    plt.xticks([])
    plt.yticks([])
    return plt
        
 
