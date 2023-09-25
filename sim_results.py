import os 
import sys
import argparse
from math import sqrt 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vals', type=int, nargs='+')
    args = parser.parse_args().vals
    score, varsum, n = args[:3], args[3:-1], args[-1]
    avg = [s/n for s in score]
    stddev = [sqrt(v/n) for v in varsum]

    for i in range(3):
        print(f'P{i+1}: mean = {avg[i]} ; stddev = {stddev[i]}')
