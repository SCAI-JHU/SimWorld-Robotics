from gym.envs.registration import register
from simworld_gym.task_generator.map_utils.base import Edge, Node, Map
from simworld_gym.task_generator.map_utils.utils.types import Vector, Road
from simworld_gym.task_generator.map_utils.config import Config


register(
    id='simworld_gym/SimpleWorld',
    entry_point='simworld_gym.envs:SimpleEnv',
    kwargs={
        'action_config': 'action_config.json'
    }
)

register(
    id='simworld_gym/BufferWorld',
    entry_point='simworld_gym.envs:BufferEnv',
    kwargs={
        'action_config': 'action_config.json',
        'buffer_max_size': 5}
)

register(
    id='simworld_gym/TrafficWorld',
    entry_point='simworld_gym.envs:TrafficEnv',

)

