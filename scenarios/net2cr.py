import os
import argparse
import subprocess
# from xml.etree import ElementTree as etree
from lxml import etree
import matplotlib.pyplot as plt

from commonroad.common.file_writer import CommonRoadFileWriter
from opendrive2lanelet.network import Network
from opendrive2lanelet.opendriveparser.parser import parse_opendrive

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object


def get_scenario_name_from_netfile(filepath:str) -> str:
    """
    Returns the scenario name specified in the net file.

    :param filepath: the path of the net file

    """
    scenario_name:str = (os.path.splitext(os.path.basename(filepath))[0]).split('.')[0]
    return scenario_name


def convert_net_to_cr(net_file:str, out_folder:str=None,verbose=False) -> str:
    """
    Converts .net file to CommonRoad xml using netconvert and OpenDRIVE 2 Lanelet Converter.

    :param net_file: path of .net.xml file
    :param out_folder: path of output folder for CommonRoad scenario.

    :return: commonroad map file
    """
    assert isinstance(net_file,str)

    if out_folder is None:
        out_folder = os.path.dirname(net_file)

    # filenames
    scenario_name = get_scenario_name_from_netfile(net_file)
    opendrive_file = os.path.join(out_folder, scenario_name + '.xodr')
    cr_map_file = os.path.join(out_folder, scenario_name + '.cr.xml')

    # convert to OpenDRIVE file using netconvert
    out = subprocess.check_output(['netconvert', '-s', net_file, '--opendrive-output', opendrive_file, '--junctions.scurve-stretch','1.0'])
    if verbose:
        print('converted to OpenDrive (.xodr)')
    # convert to commonroad using opendrive2lanelet
    # import, parse and convert OpenDRIVE file
    with open(opendrive_file, "r") as fi:
        open_drive = parse_opendrive(etree.parse(fi).getroot())

    road_network = Network()
    road_network.load_opendrive(open_drive)
    scenario = road_network.export_commonroad_scenario()
    if verbose:
        print('converted to Commonroad (.cr.xml)')
    # write CommonRoad scenario to file
    commonroad_writer = CommonRoadFileWriter(scenario, planning_problem_set=None,
                                             source="Converted from SUMO net using netconvert and OpenDRIVE 2 Lanelet Converter",
                                             tags='',author='',affiliation='')
    with open(cr_map_file, "w") as fh:
        commonroad_writer.write_scenario_to_file_io(file_io=fh)

    return scenario_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default=None, help='.net.xml file')
    args = parser.parse_args()

    if not args.net:
        print("please choose the .net.xml file")
    else:
        here = os.path.dirname(os.path.abspath(__file__)
        
        net_file = os.path.join(here, args.net)
        out_folder = os.paht.join(here, "cr")
        scenario_name = convert_net_to_cr(net_file, out_folder, True)

        # file_path = os.path.join(os.getcwd(), '/home/rong/VAE-Motion-Planning/scenarios/cr/a9.cr.xml')
        # file_path = "/home/rong/VAE-Motion-Planning/scenarios/cr/highway_20191023-2147261571838446.cr.xml"
        file_path = os.path.join(here, scenario_name, ".cr.xml")
        scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

        plt.figure(figsize=(25, 10))
        draw_object(scenario)
        # draw_object(planning_problem_set)
        plt.gca().set_aspect('equal')
        plt.show()
