from flow.controllers import IDMController
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, \
    NetParams, InitialConfig, InFlows
from flow.core.params import VehicleParams
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv, \
    ADDITIONAL_ENV_PARAMS
from flow.networks.highway import HighwayNetwork, ADDITIONAL_NET_PARAMS


def highway_example(render=None):
    sim_params = SumoParams(
        render=True,
        emission_path="./data/highway/"
        )

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {}),
        num_vehicles=20)
    vehicles.add(
        veh_id="human2",
        acceleration_controller=(IDMController, {}),
        num_vehicles=20)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="highway_0",
        probability=0.25,
        departLane="free",
        departSpeed=20)
    inflow.add(
        veh_type="human2",
        edge="highway_0",
        probability=0.25,
        departLane="free",
        departSpeed=20)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(
        inflows=inflow, additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="uniform", shuffle=True)

    network = HighwayNetwork(
        name="highway",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = LaneChangeAccelEnv(env_params, sim_params, network)

    return Experiment(env)


if __name__ == "__main__":
    exp = highway_example(render=True)
    exp.run(1, 10000, convert_to_csv=True)
