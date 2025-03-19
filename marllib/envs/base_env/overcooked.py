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
# SOFTWARE.

import gym
import numpy as np
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import pygame
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Discrete, Box

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
)

policy_mapping_dict = {
    "all_scenario": {
        "description": "overcook all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class RLlibOverCooked(MultiAgentEnv):

    def __init__(self, env_config):

        layout_mdp = OvercookedGridworld.from_layout_name(env_config["map_name"])
        core_env = OvercookedEnv.from_mdp(layout_mdp, horizon=env_config["horizon"])
        config_dict = {'base_env': core_env, 'featurize_fn': core_env.featurize_state_mdp}
        self.env = gym.make('Overcooked-v0', **config_dict)

        self.action_space = self.env.action_space
        self.observation_space = GymDict({
            "obs": Box(
            low=-self.env.observation_space.high,
            high=self.env.observation_space.high,
            dtype=self.env.observation_space.dtype)
        })
        self.episode_limit = env_config["horizon"]
        self.num_agents = 2
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        self.env_config = env_config

    def reset(self):
        o = self.env.reset()
        obs = {}
        for i, agent in enumerate(self.agents):
            obs[agent] = {
                "obs": np.float32(o["both_agent_obs"][i]),
            }
        return obs

    def step(self, action_dict):
        action_ls = []
        for agent in self.agents:
            action_ls.append(action_dict[agent])
        o, r, d, info = self.env.step(action_ls)
        rewards = {}
        obs = {}
        for i, agent in enumerate(self.agents):
            obs[agent] = {
                "obs": np.float32(o["both_agent_obs"][i]),
            }
            rewards[agent] = r

        dones = {"__all__": d}
        return obs, rewards, dones, {}

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode)

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.episode_limit,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info


if __name__ == '__main__':

    env = RLlibOverCooked({"map_name": "asymmetric_advantages", "horizon": 200})
    obs = env.reset()
    print(obs)
    # pygame.image.save(env.render(mode="rgb_array"), "./image.png")


    # def decompose_feature_vector(feature_vector, num_pots=2):
    #     """
    #     Décompose un vecteur de caractéristiques en un dictionnaire de sous-vecteurs.
        
    #     Arguments :
    #         feature_vector (np.array) : Vecteur de caractéristiques du joueur
    #         num_pots (int) : Nombre de casseroles prises en compte
        
    #     Retourne :
    #         dict : Dictionnaire contenant les sous-vecteurs
    #     """
    #     index = 0
    #     features = {}
        
    #     # Orientation (4 valeurs one-hot)
    #     features["orientation"] = feature_vector[index:index+4]
    #     index += 4
        
    #     # Objet tenu (4 valeurs one-hot)
    #     features["held_object"] = feature_vector[index:index+4]
    #     index += 4
        
    #     # Murs adjacents (4 valeurs booléennes)
    #     features["walls"] = feature_vector[index:index+4]
    #     index += 4
        
    #     # Distances aux objets clés (6 paires dx, dy)
    #     obj_names = ["onion", "tomato", "dish", "soup", "serving", "empty_counter"]
    #     for obj in obj_names:
    #         features[f"dist_{obj}"] = feature_vector[index:index+2]
    #         index += 2
        
    #     # Nombre d'ingrédients dans la soupe la plus proche
    #     features["soup_num_onions"] = feature_vector[index]
    #     index += 1
    #     features["soup_num_tomatoes"] = feature_vector[index]
    #     index += 1
        
    #     # Caractéristiques des casseroles proches
    #     for pot_idx in range(num_pots):
    #         features[f"pot_{pot_idx}_exists"] = feature_vector[index]
    #         index += 1
    #         features[f"pot_{pot_idx}_is_empty"] = feature_vector[index]
    #         index += 1
    #         features[f"pot_{pot_idx}_is_full"] = feature_vector[index]
    #         index += 1
    #         features[f"pot_{pot_idx}_is_cooking"] = feature_vector[index]
    #         index += 1
    #         features[f"pot_{pot_idx}_is_ready"] = feature_vector[index]
    #         index += 1
    #         features[f"pot_{pot_idx}_num_onions"] = feature_vector[index]
    #         index += 1
    #         features[f"pot_{pot_idx}_num_tomatoes"] = feature_vector[index]
    #         index += 1
    #         features[f"pot_{pot_idx}_cook_time"] = feature_vector[index]
    #         index += 1
    #         features[f"pot_{pot_idx}_dist"] = feature_vector[index:index+2]
    #         index += 2
        
    #     # Distances aux autres joueurs
    #     features["relative_distances_to_others"] = feature_vector[index:index+2]
    #     index += 2
        
    #     # Position absolue
    #     features["absolute_position"] = feature_vector[index:index+2]
    #     index += 2
        
    #     return features

    # # Exemple d'utilisation
    # test_vector = np.array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.,  0.,  0., -1.,
    #         2.,  0.,  0.,  0.,  0.,  2., -1.,  0.,  0.,  1.,  1.,  0.,  0.,
    #         0.,  0.,  0.,  0., -2.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,
    #         0., -2.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
    #         0.,  0., -1., -2.,  0.,  0.,  2.,  1.,  0.,  0.,  0.,  0.,  2.,
    #     -2.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  3.,  0.,
    #         1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  3., -1.,  0.,  1.,  0.,
    #         1., -5.,  1.,  6.,  2.])

    # features_dict = decompose_feature_vector(test_vector)
    # for key, value in features_dict.items():
    #     print(f"{key}: {value}")
