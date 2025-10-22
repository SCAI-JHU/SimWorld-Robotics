from gym.envs.registration import register
from simworld_gym.task_generator.map_utils.base import Edge, Node, Map
from simworld_gym.task_generator.map_utils.utils.types import Vector, Road
from simworld_gym.task_generator.map_utils.config import Config


register(
    id='gym_citynav/SimpleWorld-v0',
    entry_point='gym_citynav.envs:SimpleEnv',
    kwargs={
        'action_config': 'action_config.json'
    }
)

register(
    id='gym_citynav/BufferWorld-v0',
    entry_point='gym_citynav.envs:BufferEnv',
    kwargs={
        'action_config': 'action_config.json',
        'buffer_max_size': 5}
)

register(
    id='gym_citynav/TrafficWorld-v0',
    entry_point='gym_citynav.envs:TrafficEnv',

)

