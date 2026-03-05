from prnn.utils.predictiveNet import PredictiveNet
import os
from prnn.utils.agent import create_agent, RandomActionAgent
from prnn.utils.env import make_env
from prnn.utils.figures import IsoMapFigure
from prnn.analysis.representationalGeometryAnalysis import *
import matplotlib.pyplot as plt
import numpy as np

from prnn.analysis.representationalGeometryAnalysis import representationalGeometryAnalysis as RGA

envkey = 'MiniGrid-LRoom-18x18-v0'
envPackage = 'farama-minigrid'
action_probability = np.array([0.15,0.15,0.6,0.1])
actenc_dict = {'SpeedHD': 0, 'Velocities': 1, 'OneHot': 2}

masked_data = np.zeros((2, 7, 6))
for actenc in ['SpeedHD', 'Velocities']:#, 'OneHot']:
    for k in range(7):
        for seed in range(1, 7):
           
            env = make_env(envkey, envPackage, actenc)
            agent = RandomActionAgent(env.action_space, action_probability)

            netname = 'fig3enets/fig3e_newMasked-M_k' + str(k) + '_' + actenc + f'_S{seed}-s{seed}'
            pnet = PredictiveNet.loadNet(netname)

            rga = RGA(predictiveNet=pnet, spacemetric='cityblock', theta='mean',
                        agent=agent)
            hill_fit = rga.hill_fit['t_half']

            masked_data[actenc_dict[actenc], k, seed-1] = hill_fit

np.savez('masked_data_results.npz', masked_data=masked_data)