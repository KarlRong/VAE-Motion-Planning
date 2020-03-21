from torch.utils.data import Dataset, DataLoader
from data.MinicityDataset import TrafficDataset
from torch import nn, optim
import numpy as np
import torch

def getSampleData(model, test_data, viz_idx, device):
    X_dim = 4
    
    batch = test_data[viz_idx]
    startgoal = torch.from_numpy(batch["start_goal"]).to(device)
    occ = torch.from_numpy(batch["observation"])
    occ = occ[:, 200:600, 200:600]        
    occ = occ.unsqueeze(0)
    occ = occ.unsqueeze(1)
    adap_pool = nn.AdaptiveAvgPool3d((25,200, 200))
    occ = adap_pool(occ)
    occ = occ.to(device)
    traj = torch.from_numpy(batch["traj"]).to(device)
    print(traj.shape)
    data = torch.from_numpy(batch["data"]).to(device)
    egoid = batch["egoid"]
    print(traj.shape)
    time_stamp = batch["timeprob"]
    print("time_step: ", time_stamp)

    with torch.no_grad():
        model.eval()
        y_viz = torch.randn(1,4).to(device)
        for i in range(0, 20):
            num_viz = 12
            y_viz_p, alpha = model.inference(startgoal.expand(num_viz, X_dim * 2).to(device), traj.expand(num_viz, 25, 4),
                                    occ.expand(num_viz, 1, -1, -1, -1).to(device), num_viz)
            torch.cuda.empty_cache()
            y_viz = torch.cat((y_viz_p, y_viz), dim = 0)

    y_viz=y_viz.cpu().detach().numpy()*50
    occ=occ.cpu().detach().numpy()
    startgoal=startgoal.cpu().detach().numpy() * 50
    print("start", startgoal[0:4])
    print("goal", startgoal[4:8])

    data=data.cpu().detach().numpy() * 50
    alpha=alpha.cpu().detach().numpy()
    torch.cuda.empty_cache()

    y_viz=y_viz[:-1]
    return startgoal, egoid, occ, y_viz, time_stamp, alpha

def getDataNosample(model, test_data, viz_idx, device):
    batch = test_data[viz_idx]
    startgoal = torch.from_numpy(batch["start_goal"]).to(device)
    egoid = batch["egoid"]
    print(traj.shape)
    time_stamp = batch["timeprob"]
    print("time_step: ", time_stamp)

    startgoal=startgoal.cpu().detach().numpy() * 50
    print("start", startgoal[0:4])
    print("goal", startgoal[4:8])

    return startgoal, egoid, time_stamp

# 绘制训练数据
# test_data = test_loader.dataset
# viz_idx =   torch.randint(0,len(test_data),[1]).item()  
# #  变道场景idx
# #  
# print(viz_idx)

# batch = test_data[viz_idx]
# startgoal = torch.from_numpy(batch["start_goal"]).to(device)
# occ = torch.from_numpy(batch["observation"])
# occ = occ[:, 200:600, 200:600]        
# occ = occ.unsqueeze(0)
# occ = occ.unsqueeze(1)
# adap_pool = nn.AdaptiveAvgPool3d((25,200, 200))
# occ = adap_pool(occ)
# # adap_pool = nn.AdaptiveAvgPool3d((25,100, 600))
# # occ = adap_pool(occ)
# occ = occ.to(device)
# data = torch.from_numpy(batch["data"]).to(device)

# occ=occ.cpu().detach().numpy()
# startgoal=startgoal.cpu().detach().numpy() * 50
# data=data.cpu().detach().numpy() * 50
# torch.cuda.empty_cache()

# plotData(occ, startgoal, data)

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State as StateTupleFactory
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType, Obstacle
import numpy as np
from commonroad.geometry.shape import Polygon, ShapeGroup, Circle
from commonroad_cc.collision_detection.pycrcc_collision_dispatch import create_collision_checker, create_collision_object
import pycrcc
from scenarios.commonroad_road_boundary.construction import construct
import matplotlib.pyplot as plt
from commonroad_cc.visualization.draw_dispatch import draw_object

def show_scenario(scenario, time_step):
    plt.figure(figsize=(25, 10))
    draw_params = {}
    draw_params['time_begin'] = time_step
    draw_object(scenario, draw_params=draw_params)
    plt.autoscale()
    plt.gca().set_aspect('equal')
    plt.show()
    
def create_roadcc(scenario):
    build = ['section_triangles']
    # boundary = construct(scenario, build, ['section_triangles'], [['plot']])
    boundary = construct(scenario, build, ['section_triangles'], [])

    road_boundary_shape_list = list()
    initial_state = None
    for r in boundary['section_triangles'].unpack():
        initial_state = StateTupleFactory(position=np.array([0, 0]), orientation=0.0, time_step=0)
        p = Polygon(np.array(r.vertices()))
        road_boundary_shape_list.append(p)
    road_bound = StaticObstacle(obstacle_id=scenario.generate_object_id(), obstacle_type=ObstacleType.ROAD_BOUNDARY,
                                obstacle_shape=ShapeGroup(road_boundary_shape_list), initial_state=initial_state)

    roadcc = pycrcc.CollisionChecker()
    roadcc.add_collision_object(create_collision_object(road_bound))
    return roadcc

import math

def warp2pi(ang):
    while ang > math.pi:
        ang -= 2*math.pi
    while ang < -math.pi:
        ang += 2*math.pi
    return ang

def getproblem(scenario, egoid, time_stamp, startgoal):
    ego = scenario.obstacle_by_id(egoid)
    basepose = ego.occupancy_at_time(int(time_stamp)).shape.center

    startgoal[0] += basepose[0]
    startgoal[1] += basepose[1]
    startgoal[2] = warp2pi(startgoal[2])
    startgoal[4] += basepose[0]
    startgoal[5] += basepose[1]
    startgoal[6] = warp2pi(startgoal[6])
    
    return basepose, startgoal

def checkpath(path, checker):
    states = path.getStates()
    for state in states:
        print(state[0].getX(), state[0].getY())
        if not checker.isValid(state):
            print("valid False")

#  ompl示例
# def dynamicCarDemo(setup):
#     print("\n\n***** Planning for a %s *****\n" % setup.getName())
#     # plan for dynamic car in SE(2)
#     stateSpace = setup.getStateSpace()

#     # set the bounds for the R^2 part of SE(2)
#     bounds = ob.RealVectorBounds(2)
#     bounds.setLow(100)
#     bounds.setHigh(200)
#     stateSpace.getSubspace(0).setBounds(bounds)
#     bounds.setLow(0, -3)
#     bounds.setHigh(0, 15)
#     bounds.setLow(1, -0.52)
#     bounds.setHigh(1, 0.52)
#     stateSpace.getSubspace(1).setBounds(bounds)

#     # define start state
#     start = ob.State(stateSpace)
#     start[0] = 153
#     start[1] = 153
#     start[2] = -1.45
#     start[3] = 9.86
#     start[4] = 0

#     # define goal state
#     goal = ob.State(stateSpace)
#     goal[0] = 171.6
#     goal[1] = 123.3
#     goal[2] = -0.39
#     goal[3] = 10
#     goal[4] = 0
#     print(start, goal)
#     # set the start & goal states
#     setup.setStartAndGoalStates(start, goal, .5)
#     setup.getSpaceInformation().setStateValidityChecker(mychecker)

# #     print(setup.getSpaceInformation().getStateValidityChecker())
#     planner = oc.RRT(setup.getSpaceInformation())

    
#     setup.setPlanner(planner)
# #     print(planner.getSpaceInformation().getStateValidityChecker())
#     planner.setProblemDefinition(setup.getProblemDefinition())
# #     setup.getPlanner().setup()
#     # try to solve the problem
#     plan_res =  planner.solve(40)
#     path = planner.getProblemDefinition().getSolutionPath()
#     if plan_res:
# #         path.interpolate(); # uncomment if you want to plot the path
#         print(path.printAsMatrix())
#         if plan_res.asString() != 'Exact solution':
#             print("Solution is approximate. Distance to actual goal is %g" %
#                   planner.getProblemDefinition().getSolutionDifference())
#     else:
#         print("plan fail")
#     return path


# car = oa.DynamicCarPlanning()
# car.getSpaceInformation().setStateValidityChecker(mychecker)
# car.getSpaceInformation().setPropagationStepSize(0.3)
# car.getSpaceInformation().setStateValidityCheckingResolution(0.0001)
# path = dynamicCarDemo(car)
# plt_ompl_result(path, 0)