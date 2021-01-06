# common library
import pandas as pd
import numpy as np
import time
import gym
import os
import multiprocessing
from multiprocessing import Process, Lock

# RL models from stable-baselines
import tensorflow as tf
from stable_baselines import GAIL, SAC
from stable_baselines import ACER
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3
from stable_baselines import DQN

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv
from preprocessing.preprocessors import *
from config import config

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade



class ModelWrapper(object):
    def __init__(self, callfunc, modelname, policy, data, ext, timesteps):
        self.callfunc = callfunc
        self.model_name = modelname
        self.env = DummyVecEnv([lambda: StockEnvTrain(data)])
        self.timesteps = timesteps
        self.policy = policy
        self.ext = ext

    def run(self):
        model, modelfile = globals()[self.callfunc](self.env, self.model_name, self.policy, self.ext, self.timesteps)
        return modelfile

def par_run(wrapper):
    return wrapper.run()

def train_A2C(env_train, batchnum, model_name, timesteps=25000):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, n_steps=1 ,verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model

def train_ACER(env_train, model_name, timesteps=25000):
    start = time.time()
    model = ACER('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model

def train_DQN(env_train, model_name, policy, extension, timesteps=20000):
    """DQN model"""
    
    start = time.time()
    if extension:
        kwargs = {'double_q': True, 'prioritized_replay': True, 'policy_kwargs': dict(dueling=True)}
        model = DQN(policy, env_train, **kwargs)
    else:
        model = DQN(policy, env_train)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    modelfile = f"{config.TRAINED_MODEL_DIR}/{model_name}_{policy}"
    if extension:
        modelfile = f"{config.TRAINED_MODEL_DIR}/{model_name}_{policy}_ext"
    model.save(modelfile)
    print('Training time (DQN): ', policy, (end-start)/60,' minutes')
    return model, modelfile

def train_DDPG(env_train, model_name, policy, extension, timesteps=10000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG(policy, env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    modelfile = f"{config.TRAINED_MODEL_DIR}/{model_name}_{policy}"
    model.save(modelfile)
    print('Training time (DDPG): ', policy, (end-start)/60,' minutes')
    return model, modelfile


def train_TD3(env_train, model_name,policy, extension, timesteps=20000):
    """TD3 model"""

    # add the noise objects for TD3
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = TD3(policy, env_train, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    modelfile = f"{config.TRAINED_MODEL_DIR}/{model_name}_{policy}"
    model.save(modelfile)
    print('Training time (TD3): ', policy, (end-start)/60,' minutes')
    return model, modelfile

def train_PPO(env_train, batchnum, model_name, timesteps=50000):
    """PPO model"""

    start = time.time()
    model = PPO2('MlpPolicy', env_train, nminibatches=batchnum ,ent_coef = 0.005)
    #model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

def train_GAIL(env_train, model_name, timesteps=1000):
    """GAIL Model"""
    #from stable_baselines.gail import ExportDataset, generate_expert_traj
    start = time.time()
    # generate expert trajectories
    model = SAC('MLpPolicy', env_train, verbose=1)
    generate_expert_traj(model, 'expert_model_gail', n_timesteps=100, n_episodes=10)

    # Load dataset
    dataset = ExpertDataset(expert_path='expert_model_gail.npz', traj_limitation=10, verbose=1)
    model = GAIL('MLpPolicy', env_train, dataset, verbose=1)

    model.learn(total_timesteps=1000)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model


def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial):
    ### make a prediction based on trained model###

    ## trading env
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe


def make_env(train, rank=0):
    def _init():
        env = StockEnvTrain(train)
        env.seed(rank)
        return env

    return _init

def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []

    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    ddpgln_sharpe_list = []
    a2c_sharpe_list = []
    td3_sharpe_list = []
    td3ln_sharpe_list = []
    dqn_sharpe_list = []
    dqnln_sharpe_list = []
    dqnext_sharpe_list = []
    dqnextln_sharpe_list = []

    model_use = []

    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        end_date_index = df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window*30 + 1

        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        #historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window - 63]))]


        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])

        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        #add hy Yuqing, for parallel training
        n_envs = os.cpu_count()

        env_train = SubprocVecEnv([make_env(train, i) for i in range(n_envs)])
        
        ## validation env
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: ", 20090000, "to ",
              unique_trade_date[i - rebalance_window - validation_window])
        # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
        # print("==============Model Training===========")
        print("======A2C Training========")
        model_a2c = train_A2C(env_train, n_envs, model_name="A2C_30k_dow_{}".format(i),timesteps=30000)
        print("======A2C Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_a2c = get_validation_sharpe(i)
        print("A2C Sharpe Ratio: ", sharpe_a2c)

        print("======PPO Training========")
        model_ppo = train_PPO(env_train, n_envs, model_name="PPO_100k_dow_{}".format(i), timesteps=100000)
        print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio: ", sharpe_ppo)

        print("======DDPG, TD3 Training========")
        wrappers=[]
        wrappers.append(ModelWrapper('train_DDPG',  "DDPG_10k_dow_{}".format(i), 'MlpPolicy',train, False, 10000))
        wrappers.append(ModelWrapper('train_TD3',  "TD3_20k_dow_{}".format(i), 'MlpPolicy',train, False, 20000))
        wrappers.append(ModelWrapper('train_DDPG',  "DDPG_10k_dow_{}".format(i), 'LnMlpPolicy',train, False, 10000))
        wrappers.append(ModelWrapper('train_TD3',  "TD3_20k_dow_{}".format(i), 'LnMlpPolicy',train, False, 20000))
        # wrappers.append(ModelWrapper('train_DQN',  "DQN_20k_dow_{}".format(i), 'MlpPolicy',train, False, 20000))
        # wrappers.append(ModelWrapper('train_DQN',  "DQN_20k_dow_{}".format(i), 'LnMlpPolicy',train, False, 20000))
        # wrappers.append(ModelWrapper('train_DQN',  "DQN_20k_dow_{}".format(i), 'MlpPolicy',train, True, 20000))
        # wrappers.append(ModelWrapper('train_DQN',  "DQN_20k_dow_{}".format(i), 'LnMlpPolicy',train, True, 20000))
        pool = multiprocessing.Pool(min(len(wrappers),multiprocessing.cpu_count()))
        modelfiles = pool.map(par_run,wrappers)
        pool.close()
        pool.join()
        model_ddpg = DDPG.load(modelfiles[0])
        model_td3 = TD3.load(modelfiles[1])
        model_ddpgln = DDPG.load(modelfiles[2])
        model_td3ln = TD3.load(modelfiles[3])
        # model_dqn = DQN.load(modelfiles[4])
        # model_dqnln = DQN.load(modelfiles[5])
        # model_dqnext = DQN.load(modelfiles[6])
        # model_dqnextln = DQN.load(modelfiles[7])
        print("======DDPG Mlp Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ddpg = get_validation_sharpe(i)

        print("======TD3 Mlp Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_td3, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_td3 = get_validation_sharpe(i)
        print("======DDPG LnMlp Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ddpgln, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ddpgln = get_validation_sharpe(i)

        print("======TD3 LnMlp Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_td3ln, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_td3ln = get_validation_sharpe(i)
        # print("======DQN Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #       unique_trade_date[i - rebalance_window])
        # DRL_validation(model=model_dqn, test_data=validation, test_env=env_val, test_obs=obs_val)
        # sharpe_dqn = get_validation_sharpe(i)

        # print("======DQN Ln Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #       unique_trade_date[i - rebalance_window])
        # DRL_validation(model=model_dqnln, test_data=validation, test_env=env_val, test_obs=obs_val)
        # sharpe_dqnln = get_validation_sharpe(i)
        # print("======DQN Ext Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #       unique_trade_date[i - rebalance_window])
        # DRL_validation(model=model_dqnext, test_data=validation, test_env=env_val, test_obs=obs_val)
        # sharpe_dqnext = get_validation_sharpe(i)

        # print("======DQN Ext LN Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #       unique_trade_date[i - rebalance_window])
        # DRL_validation(model=model_dqnextln, test_data=validation, test_env=env_val, test_obs=obs_val)
        # sharpe_dqnextln = get_validation_sharpe(i)


        ppo_sharpe_list.append(sharpe_ppo)
        a2c_sharpe_list.append(sharpe_a2c)
        ddpg_sharpe_list.append(sharpe_ddpg)
        td3_sharpe_list.append(sharpe_td3)
        ddpgln_sharpe_list.append(sharpe_ddpgln)
        td3ln_sharpe_list.append(sharpe_td3ln)
        # dqn_sharpe_list.append(sharpe_dqn)
        # dqnln_sharpe_list.append(sharpe_dqnln)
        # dqnext_sharpe_list.append(sharpe_dqnext)
        # dqnextln_sharpe_list.append(sharpe_dqnextln)

        # Model Selection based on sharpe ratio
        srdict = {'PPO':sharpe_ppo, 'A2C':sharpe_a2c, 'DDPG':sharpe_ddpg, 'TD3': sharpe_td3, 'DDPGLn':sharpe_ddpgln, 'TD3Ln': sharpe_td3ln}
        modeldict = {'PPO':model_ppo, 'A2C':model_a2c, 'DDPG':model_ddpg, 'TD3': model_td3,  'DDPGLn':model_ddpgln, 'TD3Ln': model_td3ln}

        # srdict = {'PPO':sharpe_ppo, 'A2C':sharpe_a2c, 'DDPG':sharpe_ddpg, 'TD3': sharpe_td3, 'DDPGLn':sharpe_ddpgln, 'TD3Ln': sharpe_td3ln, 'DQN':sharpe_dqn, 'DQNLn': sharpe_dqnln, 'DQNEXT':sharpe_dqnext, 'DQNEXTLn': sharpe_dqnextln}
        # modeldict = {'PPO':model_ppo, 'A2C':model_a2c, 'DDPG':model_ddpg, 'TD3': model_td3,  'DDPGLn':model_ddpgln, 'TD3Ln': model_td3ln, 'DQN':model_dqn, 'DQNLn': model_dqnln, 'DQNEXT':model_dqnext, 'DQNEXTLn': model_dqnextln}
        sortdict = dict(sorted(srdict.items(), key=lambda item: item[1]))
        print(sortdict)
        modelname = list(sortdict.keys())[-1]
        model_ensemble = modeldict[modelname]
        model_use.append(modelname)

        # if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg) & (sharpe_ppo >= sharpe_):
        #     model_ensemble = model_ppo
        #     model_use.append('PPO')
        # elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
        #     model_ensemble = model_a2c
        #     model_use.append('A2C')
        # else:
        #     model_ensemble = model_ddpg
        #     model_use.append('DDPG')
        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        #print("Used Model: ", model_ensemble)
        last_state_ensemble = DRL_prediction(df=df, model=model_ensemble, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        # print("============Trading Done============")
        ############## Trading ends ##############

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
