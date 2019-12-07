def create_draw_params() -> dict:
    shape_parameters = {'opacity': 1.0,
                                'facecolor': '#FFFFFF',
                                'edgecolor': '#FFFFFF',
                                'linewidth': 0.5,
                                'zorder': 20}

    draw_params =  { 'time_begin': 200,
    #          'time_end': 10,
        'antialiased': True,
        'scenario':
                {'dynamic_obstacle':
                    {'occupancy':
                        {'draw_occupancies': -1,
                        'shape': shape_parameters
                        },
                    'shape': shape_parameters,
                    'draw_shape': True,
                    'draw_icon': False,
                    'draw_bounding_box': True,
                    'show_label': False,
                    'trajectory_steps': 40,
                    'zorder': 20
                    },
                'static_obstacle':
                    {'shape': shape_parameters},
                'lanelet_network':
                    {'lanelet':
                        {'left_bound_color': '#FFFFFF',
                        'right_bound_color': '#FFFFFF',
                        'center_bound_color': '#dddddd',
                        'draw_left_bound': True,
                        'draw_right_bound': True,
                        'draw_center_bound': False,
                        'draw_border_vertices': False,
                        'draw_start_and_direction': False,
                        'show_label': False,
                        'draw_linewidth': 0.5,
                        'fill_lanelet': False,
                        'facecolor': '#c7c7c7'
                        }
                    }
                }
        }
    # ensure that parameters are also available on higher levels
    draw_params.update(draw_params['scenario'])
    draw_params['shape'] = shape_parameters
    draw_params['shape'].update(draw_params['scenario']['static_obstacle']['shape'])
    draw_params['occupancy'] = draw_params['scenario']['dynamic_obstacle']['occupancy']
    draw_params['static_obstacle'] = draw_params['scenario']['static_obstacle']
    draw_params['dynamic_obstacle'] = draw_params['scenario']['dynamic_obstacle']
    #     draw_params['trajectory'] = draw_params['scenario']['dynamic_obstacle']['trajectory']
    draw_params['lanelet_network'] = draw_params['scenario']['lanelet_network']
    draw_params['lanelet'] = draw_params['scenario']['lanelet_network']['lanelet']
    draw_params['scenario']['lanelet'] = draw_params['scenario']['lanelet_network']['lanelet']

    return draw_params
