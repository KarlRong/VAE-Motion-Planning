import os
import argparse
import time

import matplotlib as mpl
mpl.use('TkAgg') # sets the backend for matplotlib
import matplotlib.pyplot as plt

from utils.redraw_obstacles import redraw_obstacles, set_non_blocking
from utils.create_draw_params import create_draw_params
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object


def plt_scenario(filename, nrun):
    scenario, _ = CommonRoadFileReader(filename).open()

    set_non_blocking()  # ensures interactive plotting is activated
    plt.style.use('classic')
    inch_in_cm = 2.54
    figsize = [30, 8]

    fig = plt.figure(figsize=(figsize[0] / inch_in_cm, figsize[1] / inch_in_cm))
    fig.gca().axis('equal')
    # fig.patch.set_facecolor('black')
    handles = {}  # collects handles of obstacle patches, plotted by matplotlib

    # draw_params =  create_draw_params()
    draw_params = {'time_begin':0}
    # inital plot including the lanelet network
    draw_object(scenario, handles=handles, draw_params=draw_params)
    fig.canvas.draw()
    plt.gca().autoscale()
    for handles_i in handles:
        if not handles_i:
            handles.pop()

    t1 = time.time()
    # loop where obstacle positions are updated
    for i in range(0, nrun):
        # ...
        # change positions of obstacles
        # ...
        draw_params['time_begin'] = i
        redraw_obstacles(scenario, handles=handles, figure_handle=fig, plot_limits=None, draw_params=draw_params)
        print('fps:', i/(time.time()-t1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None, help='Scenario you want to draw')
    parser.add_argument('--run', type=int, default=100, help='Scenario you want to draw')
    args = parser.parse_args()

    if not args.file:
        print("A .cr.xml file need to be chose. Default running")
        file_path = os.getcwd() + '/commonroad_data/USA_Lanker-1_1_T-1.xml'
    else:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  args.file)

    plt_scenario(file_path, args.run)
