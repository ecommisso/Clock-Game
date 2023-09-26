import os 
import sys
import argparse
from math import sqrt 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vals', type=int, nargs='+')
    args = parser.parse_args().vals
    n = args[-1]
    scores = [[],[],[]]
    for i, s in enumerate(args[:-1]):
        scores[i%3].append(s)
    for s in scores:
        print(s)
    avg = [sum(s)/n for s in scores]
    stddev = [sqrt(sum(map(lambda x: (x - avg[i])**2, s)) / n) for i, s in enumerate(scores)]

    for i in range(3):
        print(f'P{i+1}: mean = {avg[i]} ; stddev = {stddev[i]}')
