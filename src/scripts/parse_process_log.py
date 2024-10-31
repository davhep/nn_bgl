#!/usr/bin/python3
import re
import numpy as np
import pandas as pd
import sys
import json
import time
import _pickle as pickle
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_log_file(filename):

    weights = defaultdict(dict)
    result_train = {}
    result_validate = {}
    f = open(filename, 'r')
    for row in f:
        row = row.split(' ')
        if (row[0] == 'weights'):
            weights[(int(row[2]),int(row[3]))].update({int(row[1]): float(row[4])})

        if (row[1] == 'epoch'):
            result_train[int(row[2])] = float(row[7])
            result_validate[int(row[2])] = float(row[11])

    plt.plot(*zip(*sorted(result_train.items())))
    plt.plot(*zip(*sorted(result_validate.items())))
    plt.savefig("png/rez.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    for edge in weights:
        plt.plot(*zip(*sorted(weights[edge].items())))
        i, o = edge
        print(i ,o)
        fig_name="png/"+str(i)+"_"+str(o)+".png"
        plt.savefig(fig_name, format="png", bbox_inches="tight")
        plt.clf()

filename = sys.argv[1]
parse_log_file(filename)
