# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.sys._get

import inspect
import time

from typing import Dict, Union
from CybORG.Tests.utils import CustomGenerator
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Discrete, Box
import supersuit as ss
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from gym import spaces
from CybORG import CybORG
from CybORG.Simulator.Scenarios import DroneSwarmScenarioGenerator
from CybORG.Agents.Wrappers import PettingZooParallelWrapper


def create_env(environment: str = "sim", env_config=None, agents: dict = None,
               seed: Union[int, CustomGenerator] = None,
               drone_swarm_scenario_configuration: Dict = {}) -> PettingZooParallelWrapper:
    sg = DroneSwarmScenarioGenerator(**drone_swarm_scenario_configuration)
    env = CybORG(scenario_generator=sg, environment=environment,
                 env_config=env_config, agents=agents, seed=seed)
    return PettingZooParallelWrapper(env)


REGISTRY = {}
REGISTRY["cage3"] = create_env

policy_mapping_dict = {
    "cage3": {
        "description": "CybORG 3rd CAGE Challenge",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    }
}


class RLlibCBG(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)
        self.env = REGISTRY[map](**env_config)

        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions
        # env = ss.pad_observations_v0(env)
        # env = ss.pad_action_space_v0(env)

        # self.env = ParallelPettingZooEnv(env)

        observation_space = self.env.observation_spaces[self.env.agents[0]]
        self.i = 0
        self.action_space = spaces.Discrete(
            self.env.action_spaces[self.env.agents[0]].n)

        # if GymSpace is MultiDiscrete, then we need to convert it to Box
        self.observation_space = GymDict({"obs": Box(
            low=0,
            high=observation_space.nvec.max(),
            shape=(observation_space.nvec.shape[0],),
            dtype=observation_space.dtype)})

        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        self.i = 0
        original_obs = self.env.reset()
        obs = {}
        for i in self.agents:
            obs[i] = {"obs": original_obs[i]}
        return obs

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = r[key]
            obs[key] = {
                "obs": o[key]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info

    def render(self, mode="human"):
        # time.sleep(0.25)
        return self.env.render(mode=mode)

    def seed(self, seed=None):
        self.env.seed(seed)

    def close(self):
        self.env.close()

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 30,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
