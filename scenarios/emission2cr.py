import os
import csv
from collections import defaultdict

import numpy as np
import math

cwd = os.getcwd()
emission_path =  os.path.join(
    cwd, 
    "scenarios/data/highway_20191023-1757041571824624.5293925-emission.csv")
cr_path = os.path.join(
    cwd,
    "scenarios/cr/highway_20191023-2147261571838446.cr.xml"
)

print(emission_path)