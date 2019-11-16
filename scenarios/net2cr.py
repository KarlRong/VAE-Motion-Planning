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
    scenario_name:str = (os.path.splitext(os.path.basename(filepath))[0]).split('.')[0]
    return scenario_name


def convert_net_to_cr(net_file:str, out_folder:str=None,verbose=False) -> str:
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
    commonroad_writer = CommonRoadFileWriter(
        scenario, 
        planning_problem_set=None,
        source="Converted from SUMO net using netconvert and OpenDRIVE 2 Lanelet Converter",
        tags='',
        author='',
        affiliation=''
        )
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
        here = os.path.dirname(os.path.abspath(__file__))
        net_file = os.path.join(here, args.net)
        out_folder = os.path.join(here, "./commonroad_data")
        scenario = convert_net_to_cr(net_file, out_folder, True)

        file_path = os.path.join(out_folder, scenario +  ".cr.xml")

        from utils.show_scenario import show_scenario
        show_scenario(file_path)

