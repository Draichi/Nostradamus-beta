import ray
import collections
import os
import pickle
import pandas as pd
from utils import get_datasets
from ray.tune import grid_search, run
from core_env import TradingEnv
from ray.tune import grid_search, run_experiments
from ray.tune.registry import register_env

class Nostradamus:

    def __init__(self, assets=['BTC', 'LTC', 'ETH'], currency='USDT', granularity='day', datapoints=600):

        self.assets = assets
        self.currency = currency
        self.granularity = granularity
        self.datapoints = datapoints
        self.df = {}
        self.config_spec = {}
        self.check_variables_integrity()
        self.populate_dfs()

    def check_variables_integrity(self):
        if type(self.assets) != list or len(self.assets) == 0:
            raise ValueError("Incorrect 'assets' value")
        if type(self.currency) != str:
            raise ValueError("Incorrect 'currency' value")
        if type(self.granularity) != str:
            raise ValueError("Incorrect 'granularity' value")
        if type(self.datapoints) != int or 1 > self.datapoints > 2000:
            raise ValueError("Incorrect 'datapoints' value")

    def populate_dfs(self):
        for asset in self.assets:
            self.df[asset] = {}
            self.df[asset]['train'], self.df[asset]['rollout'] = get_datasets(asset=asset,
                                                                              currency=self.currency,
                                                                              granularity=self.granularity,
                                                                              datapoints=self.datapoints)

    def generate_config_spec(self, lr_schedule, df_type):
        self.config_spec = {
            "lr_schedule": grid_search(lr_schedule),
            "env": "YesMan-v1",
            "num_workers": 3,  # parallelism
            'observation_filter': 'MeanStdFilter',
            'vf_share_layers': True,
            "env_config": {
                'assets': self.assets,             # *         no rollout mandar o config do TradingEnv(config)
                'currency': self.currency,         # *         populado com esse env_config aqui
                'granularity': self.granularity,
                'datapoints': self.datapoints,
                'df_complete': {},
                'df_features': {}
            },
        }
        self.add_dfs_to_config_spec(df_type=df_type)

    def add_dfs_to_config_spec(self, df_type):
        for asset in self.assets:
            self.config_spec['env_config']['df_complete'][asset] = self.df[asset][df_type]
            self.config_spec['env_config']['df_features'][asset] = self.df[asset][df_type].loc[:,
                                                                                               self.df[asset][df_type].columns != 'Date']

    def find_results_folder(self):
        return os.getcwd() + '/results'

    def trial_name_string(self, trial):
        return str('trial_name_string_from_line_78')

    def train(self, algo='PPO', timesteps=3e10, checkpoint_freq=100, lr_schedule=[[[0, 7e-5], [3e10, 7e-6]]]):
        register_env("YesMan-v1", lambda config: TradingEnv(config))
        ray.init()

        self.generate_config_spec(lr_schedule=lr_schedule, df_type='train')

        run(name="experiment_name",
            run_or_experiment=algo,
            stop={'timesteps_total': timesteps},
            checkpoint_freq=checkpoint_freq,
            config=self.config_spec,
            local_dir=self.find_results_folder(),
            trial_name_creator=self.trial_name_string)
