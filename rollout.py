#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
import pickle

import gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.registry import register_env
from ray.tune.util import merge_dicts


# > Note: if you use any custom models or envs, register them here first, e.g.:
# $ ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
# ! register_env("pa_cartpole", lambda _: ParametricActionCartpole(10))



# @ esse metodo tem que ir junto para core_main
def get_instruments_from_checkpoint(checkpoint):
    config = {}
    # Load configuration from file
    config_dir = os.path.dirname(checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or "
            "its parent directory.")
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    if config['env_config']:
        env_config = config['env_config']
        if env_config['assets']:
            assets = env_config['assets']
        else:
            raise ValueError('assets does not exists in env_config')
        if env_config['currency']:
            currency = env_config['currency']
        else:
            raise ValueError('currency does not exists in env_config')
        if env_config['datapoints']:
            datapoints = env_config['datapoints']
        else:
            raise ValueError('datapoints does not exists in env_config')
        if env_config['granularity']:
            granularity = env_config['granularity']
        else:
            raise ValueError('granularity does not exists in env_config')
        if env_config['variables']:
            variables = env_config['variables']
        else:
            raise ValueError('variables does not exists in env_config')
    else:
        raise ValueError('env_config does not exists in params.pkl')
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
    return config, assets, currency, datapoints, granularity, variables

class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def rollout(agent, env_name, num_steps, no_render=True):
    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "workers"):
        print('is multiagent')
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: _flatten_action(m.action_space.sample())
            for p, m in policy_map.items()
        }
    else:
        print('is not multiagent')
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    steps = 0
    while steps < (num_steps or steps + 1):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = _flatten_action(a_action)  # tuple actions
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, _ = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if not no_render:
                env.render()
            steps += 1
            obs = next_obs
        print("Episode reward", reward_total)

if __name__ == "__main__":
    from core_env import TradingEnv
    from utils import get_datasets

    checkpoint_path = '/home/lucas/Documents/new_nostradamus/results/teste_do_rollout/1_2019-10-05_20-45-58nxzjv1tc/checkpoint_10/checkpoint-10'
    agent_config, assets, currency, datapoints, granularity, variables = get_instruments_from_checkpoint(checkpoint_path)
    
    # > isso tudo tem que vir em core_main.py
    config = {
        'assets': assets,
        'currency': currency,
        'granularity': granularity,
        'datapoints': datapoints,
        'df_complete': {},
        'df_features': {},
        'variables': variables
    }

    df = {}
    for asset in assets:
        df[asset] = {}
        _, df[asset]['rollout'] = get_datasets(asset=asset,
                                               currency=currency,
                                               granularity=granularity,
                                               datapoints=datapoints)

    for asset in assets:
        config['df_complete'][asset] = df[asset]['rollout']
        config['df_features'][asset] = df[asset]['rollout'].loc[:, df[asset]['rollout'].columns != 'Date']

    env_name = 'YesMan-v1'

    register_env(env_name, lambda config: TradingEnv(config))
    ray.init()
    cls = get_agent_class('PPO')                          # ? pq cls fica dessa cor ?
    agent = cls(env=env_name, config=agent_config)
    agent.restore(checkpoint_path)
    
    num_steps = int(len(config['df_complete'][assets[0]]))
    no_render = False

    rollout(agent, env_name, num_steps, no_render)

    # $ esse codigo esta indo bem até o core_render lá tem um quit()