# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/01/10 14:23:08
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import numpy as np
import matplotlib.pyplot as plt


def plot_pr_curve(save_file, prs, rcs, title="PR Curve", label=None):
    """
    Plot precision-recall curve

    Args:
        save_file (str): save file path
        prs (list): precision list
        rcs (list): recall list
        title (str): title
        legend (list): legend

    Return:
        None
    """
    plt.figure()
    plt.axis("equal")
    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.yticks(np.arange(0, 1.0, step=0.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.plot(rcs, prs, "r-", label=label)
    plt.rc("legend", fontsize=10)
    plt.legend()

    f_scores = np.linspace(0.2, 0.9, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color=[0, 0.5, 0], alpha=0.3)
        plt.annotate(
            "f={0:0.1}".format(f_score),
            xy=(0.9, y[45] + 0.02),
            alpha=0.4,
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_file)
    plt.savefig(os.path.splitext(save_file)[0] + ".pdf")
    plt.close()


def calc_AP(prs, rcs):
    """
    calculate AP

    Args:
        prs (np.array): precision array
        rcs (np.array): recall array
    Return:
        AP (float)
        P (float)
        R (float)
    """
    precision = np.concatenate(([0.0], prs))
    recall = np.concatenate(([0.0], rcs))

    i = np.where(recall[1:] != recall[:-1])[0]
    AP = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
    P, R = prs[-1], rcs[-1]

    return AP, P, R
