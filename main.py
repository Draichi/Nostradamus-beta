if __name__ == '__main__':
    from utils import random_emojis
    random_emojis()
    from core_main import Nostradamus
    env = Nostradamus(assets=['OMG','BTC','ETH'],
                      currency='USDT',
                      granularity='day',
                      datapoints=600)

    env.train(timesteps=5e5,
              checkpoint_freq=30,
              lr_schedule=[
                  [
                      [0, 7e-5],  # [timestep, lr]
                      [100, 7e-6],
                  ],
                  [
                      [0, 6e-5],
                      [100, 6e-6],
                  ]
              ],
              algo='PPO')
    # env.backtest(checkpoint_file='logs/agora_vai/PPO_YesMan-v1_1_lr_schedule=_2019-08-10_22-07-544zdnasyx/checkpoint_20/checkpoint-20')
