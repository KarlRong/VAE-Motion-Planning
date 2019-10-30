import os
import argparse
import matplotlib.pyplot as plt

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object


def draw_scenario(file_path):
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    plt.figure(figsize=(250, 100))
    draw_object(scenario)
    draw_object(planning_problem_set)
    plt.gca().set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None, help='Scenario you want to draw')
    args = parser.parse_args()

    if not args.file:
        print("A .cr.xml file need to be chose")
    else:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  args.file)

        draw_scenario(file_path)

