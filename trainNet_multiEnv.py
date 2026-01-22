from prnn.utils.data import generate_trajectories, create_dataloader, mergeDatasets
from prnn.utils.env import make_env
from prnn.utils.agent import RandomActionAgent, RandomHDAgent
from prnn.utils.predictiveNet import PredictiveNet
from prnn.utils.general import saveFig
import argparse
import wandb

#TODO: get rid of these dependencies
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os

import sys
sys.path.append('../')

def list_of_strings(arg):
    return arg.split(',')


from scipy.spatial.distance import pdist, squareform
from sklearn import manifold
from scipy.stats import spearmanr
from prnn.utils.plotUtils import setNiceAxes

class multiEnvGeometryAnalysis:
    def __init__(self, pN, envs,
                 timesteps_wake=10000, 
                 timesteps_sleep=250, numsleeps = 25, sleepMeanStd = (0,0.1),
                 replayThresh = 0.1, theta='expand',
                 withIsomap=False, withIsomap3d=False):
        
        #try 2 remove necessity to save
        self.pN = pN
        self.envs = envs

        #The Random action agent, used for exploring the environment
        action_probability = np.array([0.15,0.15,0.6,0.1])
        agent = RandomActionAgent(envs[0].action_space,action_probability)
        
        #Run a trial in each environment
        self.WAKEactivity = []
        for env in envs:
            self.WAKEactivity.append(runWAKE(pN, env, agent, timesteps_wake, theta=theta))
            
        #Calculate the manifold distance between the two environments (TODO modify for >2)
        self.manifoldDistance = calculateManifoldDistance(self.WAKEactivity[0]['h'],self.WAKEactivity[1]['h'])
        self.manifoldDistance_obs = calculateManifoldDistance(self.WAKEactivity[0]['obs_unique'],
                                                              self.WAKEactivity[1]['obs_unique'])


        # #Fit the isomap (visualization)
        # self.withIsomap = withIsomap
        # if withIsomap:
        #     self.isomap = fitIsomap([self.WAKEactivity[0]['h'],self.WAKEactivity[1]['h']],
        #                             points_per_map=4000, n_neighbors=20)
        # self.withIsomap3d = withIsomap3d
        # if withIsomap3d:
        #     self.isomap3d = fitIsomap([self.WAKEactivity[0]['h'],self.WAKEactivity[1]['h']],
        #                             points_per_map=4000, n_neighbors=20, n_components=3)

        # NEED TO ACCOUNT FOR >2 ENVIRONMENTS !!!!!
        acts = [a["h"] for a in self.WAKEactivity]
        if withIsomap:
            self.isomap = fitIsomap(acts, points_per_map=4000, n_neighbors=20)

        if withIsomap3d:
            self.isomap3d = fitIsomap(acts, points_per_map=4000, n_neighbors=20, n_components=3)


        #Calculate spatial RSA (sRSA) in each envirionment
        self.sRSA = []
        self.sRSA_dists = []
        for activity in self.WAKEactivity:
            sRSA = calculateRSA_space(activity)
            self.sRSA.append(sRSA[0][0])
            self.sRSA_dists.append((sRSA[1],sRSA[2]))

        #Run the sleep epochs and calculate sleep metrics
        self.SLEEPactivity = []
        if timesteps_sleep>0 and numsleeps>0:
            self.replayThresh = replayThresh
            print(f'Running Sleeps ({numsleeps})')
            for s in range(numsleeps):
                self.SLEEPactivity.append(runSLEEP(pN, sleepMeanStd[0], sleepMeanStd[1],
                                                   timesteps_sleep,suppressPrint=True))
            
            print('Calculating Sleep-Wake Distances')
            self.SWdist = np.zeros((len(self.WAKEactivity),len(self.SLEEPactivity)))
            self.SWdists = np.zeros((len(self.WAKEactivity),len(self.SLEEPactivity),timesteps_sleep))
            for sIdx, Sactivity in enumerate(self.SLEEPactivity):
                for wIdx, Wactivity in enumerate(self.WAKEactivity):
                    SW = calculateManifoldDistance(Wactivity['h'],Sactivity['h'])
                    self.SWdist[wIdx,sIdx]=SW[0]
                    self.SWdists[wIdx,sIdx,:]=SW[2]
            self.isReplay = self.SWdist < replayThresh
    

    def SpatialGeometryFigure(self, exsleep = 0, netname=None, savefolder=None, show=False):

        fg = plt.figure()
        plt.subplot(3,2,1)
        trainingPanel_sRSA(self.pN)

        plt.subplot(3,2,5)
        trainingPanel_dist(self.pN)
        
        plt.subplot(3,4,3)
        acts = [a["h"] for a in self.WAKEactivity]
        isomapPanel(self.isomap,acts)

        plt.subplot(3,4,4, projection='3d')
        isomapPanel(self.isomap3d,acts)

        for eidx, env_sRSA in enumerate(self.sRSA_dists):
            plt.subplot(3,4,7+eidx)
            sRSApanel(env_sRSA, SWdist=self.SWdists[eidx,:,:])
            plt.title('Env'+str(eidx))
        plt.gca().set_yticklabels([])
        plt.ylabel('')

        plt.subplot(4,4,16)
        self.sleepReplayPanel(exsleep)

        if netname is not None:
            saveFig(fg,'SpatialGeometry_'+netname,savefolder,
                    filetype='png',dpi=300)
        if show:
            plt.show()
        return fg
    
    def IsomapOnlyFigure(self, show=False, maxplotpoints=10000, rotate=(0,0)):
        fg = plt.figure(figsize=(10,4))

        # 2D
        ax1 = fg.add_subplot(1, 2, 1)
        isomapPanel(self.isomap,
                [self.WAKEactivity[0]['h'], self.WAKEactivity[1]['h']],
                maxplotpoints=maxplotpoints,
                rotate=rotate)
        ax1.set_title("Isomap 2D")

        # 3D
        ax2 = fg.add_subplot(1, 2, 2, projection='3d')
        isomapPanel(self.isomap3d,
                [self.WAKEactivity[0]['h'], self.WAKEactivity[1]['h']],
                maxplotpoints=maxplotpoints,
                rotate=rotate)
        ax2.set_title("Isomap 3D")

        if show:
            plt.show()
        return fg

    def sleepReplayPanel(self, exsleep):
        plt.plot(self.SWdists[0,exsleep,:],self.SWdists[1,exsleep,:],
                 '.',color='grey', markersize=1)
        plt.plot(self.SWdist[0,exsleep],self.SWdist[1,exsleep],'r*')
        plt.plot(self.SWdist[0,:],self.SWdist[1,:],'r.')
        plt.plot([self.replayThresh,self.replayThresh],plt.xlim(),'k--')
        plt.plot(plt.ylim(),[self.replayThresh,self.replayThresh],'k--')
        ax = plt.gca()
        ax.axvspan(0,self.replayThresh, color='blue' , alpha=0.1)
        ax.axhspan(0,self.replayThresh, color='orange' , alpha=0.1)
        plt.xlabel('S-W Dist, Env0')
        plt.ylabel('S-W Dist, Env1')
        setNiceAxes() 
    

def sRSApanel(sRSA_dists, SWdist = None,
              vmin=None, vmax=None):
        dists = sRSA_dists[0]
        sp_dists = sRSA_dists[1]
        goodmax=1
        (hist2,rbins,sbins) = np.histogram2d(dists, sp_dists, 
                                            bins=[np.linspace(0,goodmax,50),
                                                np.arange(-0.5,16.5,1)])
        hist2 = hist2/np.sum(hist2,axis=0)

        plt.imshow((hist2), origin='lower', aspect='auto',
                  extent=(sbins[0],sbins[-1],rbins[0],rbins[-1]),
                  cmap='binary', vmin=vmin, vmax=vmax,
                  interpolation='none')

        if SWdist is not None:
            n,bins = np.histogram(SWdist,bins=rbins)
            n = n/np.sum(n)
            plt.imshow(np.expand_dims(n,axis=1),
                        origin='lower', aspect='auto',
                        extent=(-2.5,-1.5,rbins[0],rbins[-1]),
                        cmap='binary')
            plt.xlim([-2.5,sbins[-1]])
            
        plt.xlabel('Spatial Dist')
        plt.ylabel('Neural Dist')
        setNiceAxes() 

def trainingPanel_sRSA(pN):
    ax = plt.gca()
    TrainMetrics = pN.TrainingSaver.drop(columns="loss")
    TrainMetrics = TrainMetrics.dropna()

    #This needs to be automated from TrainingSaver
    env0time = [0,30000]
    env1time = [30000,60000]

    plt.plot(TrainMetrics.sRSA_env0,'o')
    plt.plot(TrainMetrics.sRSA_env1,'o')
    plt.xlabel('Train Step')
    plt.ylabel('sRSA')
    #plt.legend(['Env0','Env1'])
    #ax.axvspan(env0time[0], env0time[1], color='blue' , alpha=0.1)
    #ax.axvspan(env1time[0], env1time[1], color='orange' , alpha=0.1)
    setNiceAxes()  

def trainingPanel_dist(pN, bothEnvs=True, niceAxes=True):
    ax = plt.gca()
    TrainMetrics = pN.TrainingSaver.drop(columns="loss")
    TrainMetrics = TrainMetrics.dropna()

    #This needs to be automated from TrainingSaver
    env0time = [0,30000]
    env1time = [30000,60000]

    plt.plot(TrainMetrics.D_01,'o')
    if bothEnvs:
        plt.plot(TrainMetrics.D_10,'o')
    plt.xlabel('Train Step')
    plt.ylabel('Manifold Dist')
    #ax.axvspan(env0time[0], env0time[1], color='blue' , alpha=0.1)
    #ax.axvspan(env1time[0], env1time[1], color='orange' , alpha=0.1)
    if niceAxes:
        setNiceAxes() 

        

## FUNCTIONS ##        
def runWAKE(pN, env, agent, timesteps_wake,theta='expand'):

    print('Running WAKE')
    a = {}
    a['obs_env'],a['act_env'],a['state'],_ = pN.collectObservationSequence(env,
                                                         agent,
                                                         timesteps_wake,
                                                          obs_format=None)
    a['obs'],a['act'] = env.env2pred(a['obs_env'],a['act_env'])
    a['obs_pred'], a['obs_next'], h = pN.predict(a['obs'],a['act'])

    #Get the observations from all unique positions, in same form as h
    fullstate = np.vstack((a['state']['agent_pos'].transpose(), a['state']['agent_dir']))
    _,indices = np.unique(fullstate,return_index=True,axis=1)
    a['obs_unique'] = np.squeeze(a['obs'][:,indices,:].detach().numpy())

    #Deal with rollout dimension 
    if theta == 'mean':
        h = h.mean(axis=0,keepdims=True)
        a['act'] = a['act'][:,:h.size(dim=1),:]
        a['state']['agent_pos'] = a['state']['agent_pos'][:h.size(dim=1)+1,:]
        a['state']['agent_dir'] = a['state']['agent_dir'][:h.size(dim=1)+1]
    if theta == 'expand':
        k = h.size(dim=0)
        h = h.transpose(0,1).reshape((-1,1,h.size(dim=2))).swapaxes(0,1)
        a['state']['agent_pos'] = np.repeat( a['state']['agent_pos'], k, axis=0)
        a['state']['agent_pos'] = a['state']['agent_pos'][:h.size(dim=1)+1,:]

    a['h'] = np.squeeze(h.detach().numpy())
    return a


def runSLEEP(pN, noisemag, noisestd, timesteps_sleep, suppressPrint=False):
    if not suppressPrint: print('Running SLEEP')
    a = {}
    a['obs_pred'],h_t,a['noise_t'] = pN.spontaneous(timesteps_sleep,
                                               noisemag,
                                               noisestd)
    a['h'] = np.squeeze(h_t.detach().numpy())
    return a


def randSubSample(h, maxN=3000, axis=0):
    #pick random N of timesteps if bigger than maxN timesteps
    nT = np.size(h,axis)
    randIDX = np.arange(nT)
    if nT > maxN:
        randIDX = np.random.randint(0, high=nT, size=maxN)
        h = h[randIDX,:]
    return h, randIDX


def calculateManifoldDistance(A,B, metric='cosine'):
    #Point Manifolds: A,B [time x neurons]
    #Output: ( D(A<-B), D(B<-A), D(A<-j), D(B<-i) )
    #Random Subsample of timepoints (reduce compute/RAM usage)
    A, keepIDX_A = randSubSample(A, axis=0)
    B, keepIDX_B = randSubSample(B, axis=0)
    
    #Calcualte distance between all pairs of points
    X = np.concatenate((A,B))
    ndists = squareform(pdist(X,metric))
    d_ij = SWdist = ndists[:A.shape[0],A.shape[0]:]
    
    #For points in each manifold, find distance to closest point in other manifold
    d_Aj = np.min(d_ij,axis=0)
    d_iB = np.min(d_ij,axis=1)
    #Calculate median for manifold distance
    D_AB = np.median(d_Aj)
    D_BA = np.median(d_iB)
    
    return (D_AB,D_BA,d_Aj,d_iB)


def fitIsomap(activities, n_neighbors=30, n_components=2, points_per_map=3000):
    #activities = [list of [time x neurons] arrays]
    print('Fitting Isomap')
    #X = self.WAKEactivity['h']
    #h_wake = WAKEactivity['h']
    #h_sleep = SLEEPactivity['h']
    
    activities = [randSubSample(i, maxN=points_per_map, axis=0)[0] for i in activities]
    X = np.concatenate(np.array(activities))
    
    method = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, metric='cosine')
    method.fit(X)
    return method

def calculateNeuralDistWAKE(WAKEactivity_h, metric='cosine'):
    h_np, keepIDX = randSubSample(WAKEactivity_h, axis=0)

    dists = pdist(h_np,metric)
    return dists, keepIDX

def calculateSpatialDist(WAKEactivity_state, keepIDX=None, metric='euclidian'):
    position = WAKEactivity_state['agent_pos'][:-1,:]
    if keepIDX is not None:
        position = position[keepIDX,:]

    sp_dists = pdist(position,metric)
    return sp_dists

def calculateRSA_space(WAKEactivity, neuralmetric='cosine',
                      usecells = None, spacemetric='euclidean'):
    dists,keepIDX = calculateNeuralDistWAKE(WAKEactivity['h'], 
                                                 neuralmetric)
    sp_dists = calculateSpatialDist(WAKEactivity['state'],keepIDX,
                                        metric=spacemetric)

    RSA = spearmanr(dists,sp_dists)

    return (RSA,dists,sp_dists)



## PLOTTING FUNCTIONS ##

def isomapPanel(isomap, activities,colorvar='environment', onsetTransient=10, mapcenter=[18,18],maxplotpoints=10000,
               rotate=(0,0)):
    maxenvplotpoints = int(maxplotpoints/len(activities))
    activities = [randSubSample(i, maxN=maxenvplotpoints,axis=0)[0] for i in activities]
    if colorvar == 'environment':
        color = [i*np.ones_like(a[:,0]) for i,a in enumerate(activities)]
        X = np.concatenate(np.array(activities))
        color = np.concatenate(np.array(color))
    elif colorvar == 'position':
        state = self.WAKEactivity['state']
        color = np.arctan((state['agent_pos'][:-1,0]-mapcenter[0])/(state['agent_pos'][:-1,1]-mapcenter[1]))
    elif colorvar == 'SleepWake':
        h_sleep = self.SLEEPactivity['h'][onsetTransient:,:]
        color = np.concatenate((np.ones((X.shape[0])),np.zeros((h_sleep.shape[0]))))
        X = np.concatenate((X,h_sleep))
    elif colorvar == 'Sleep':
        h_sleep = self.SLEEPactivity['h'][onsetTransient:,:]
        color = np.tile([0.7,0,0],(np.size(h_sleep,axis=0),1))
        X = h_sleep
    elif colorvar == 'HD':
        state = self.WAKEactivity['state']
        color = state['agent_dir'][:-1]
    elif colorvar == 'action':
        color,_ = self.getActionIDs()

    Y = isomap.transform(X)

    ax = plt.gca()
    if Y.shape[1]==2:
        ax.scatter(Y[:, 0], Y[:, 1], 
                   c=color, marker='.', s=4)
        #ax.view_init(rotate[0]-140, rotate[1]+60)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
    elif Y.shape[1]==3:
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], 
                   c=color, marker='.', s=4)
        ax.view_init(rotate[0]-140, rotate[1]+60)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
    #ax.axis('off')
    #ax.zaxis.set_ticklabels([])
    #ax.set_title(colorvar)
    return


# Parse arguments
parser = argparse.ArgumentParser()

## pRNN Parameters
parser.add_argument("--pRNNtype", default='thcycRNN_5win_full',
                    help="which pRNN (Default: thcycRNN_5win_full)")
parser.add_argument("--losstype", default='predMSE',
                    help="Loss function (Default: predRMSE)")
parser.add_argument("--hiddensize", default=500, type=int,
                     help="how many hidden units? (Default: 300")
parser.add_argument("--ntimescale", default=2, type=float,
                     help="Neural timescale (Default: 2 timesteps)")
parser.add_argument('--trainBias', action='store_true', default=True)
parser.add_argument('--identityInit', action='store_true', default=False)
parser.add_argument("--act_enc", default='SpeedHD',
                     help="Action encoding, options: OnehotHD, SpeedHD (default), Onehot, Velocities")
# parser.add_argument("--lambda_context", default=1, type=float,
#                         help="Lambda for context input to sparse layer (Default: 1)")
# parser.add_argument("--lambda_direct", default=1, type=float,
#                         help="Lambda for direct input to RNN layer (Default: 1)")
# parser.add_argument("--lambda_sparse", default=1, type=float,
#                         help="Lambda for sparse input to RNN layer (Default: 1)")
# parser.add_argument("--sparse_beta", default=1, type=float,
#                         help="Softmax temperature for sparse input to RNN layer (Default: 1)")
# parser.add_argument("--sparse_size", default=1500, type=int,
#                         help="Size of sparse layer (Default: 1500)")

## Environment parameters
parser.add_argument("--envlist", type=list_of_strings,
                    default=['MiniGrid-LRoom-18x18-v0','MiniGrid-TRoom-20x20-v0','MiniGrid-DonutRoom-16x16-v0'],
                    help="List of environments to train on")
parser.add_argument("--env_package", default='farama-minigrid',
                    help="Environment Package")
parser.add_argument("--trainCurriculum", default='mixed-batch',
                    help="Multi-Env training curriculium")
parser.add_argument("--epochsPerEnv", default=60, type=int,
                    help="How many epochs is each environment trained on")
parser.add_argument("--seq_length", default=500, type=int,
                     help="how long is each behavioral sequence? (Default: 1000")
parser.add_argument("--trials_per_epoch", default=500, type=int,
                     help="many trials in an epoch? (Default: 500")
parser.add_argument("--batch_size", default=16, type=int,
                     help="many trials in an minibatch? (Default: 16")

## Dataset Parameters
parser.add_argument("--datasetSize", default=10000, type=int,
                    help="How big of a dataset to pre-compute")

## File Management
parser.add_argument("--datasetfolder", default=os.environ["TMPDIR"],
                    help="Where to save the dataset?")
parser.add_argument("--netsfolder", default='MultiEnv',
                    help="Parent folder to save the net?")
parser.add_argument("--savefolder", default='',
                    help="Subfolder to save the net? (foldername/)")
parser.add_argument("--loadfolder", default='',
                    help="Where to load the net? (foldername/)")
parser.add_argument("--namext", default='',
                     help="Extension to the savename?")
parser.add_argument("-c", "--contin", action="store_true",
                     help="Continue previous training?")
parser.add_argument("--load_env", default=-1, type=int,
                     help="Load Environment for continued Training. Specify unique env id")
parser.add_argument('--saveTrainData', action='store_true', default=False)
parser.add_argument('--no-saveTrainData', dest='saveTrainData', action='store_false')

## Hyperparameters
parser.add_argument("-s", "--seed", default=8, type=int,
                     help="Random Seed? (Default: 8)")

parser.add_argument("--lr", default=3e-3, type=float,    #former default:2e-4 (not relative)
                     help="Learning Rate? (Relative to init sqrt(1/k) for each layer) (Default: 1e-3)")
parser.add_argument("--bias_lr", default=1, type=float,    #former default:2e-4 (not relative)
                     help="Bias Learning Rate? (Relative to learning rate) (Default: 1)")
parser.add_argument("--weight_decay", default=3e-3, type=float, #former default:6e-7 (not relative)
                     help="Weight Decay? (Relative to learning rate) (Default: 0)")
parser.add_argument("--bptttrunc", default=1e8, type=int,
                     help="BPTT Truncation window? (Default: 1e8 (~Inf))")
parser.add_argument("--dropout", default=0.15, type=float,
                     help="Dropout probability (Default: 0)")
parser.add_argument("--noisemean", default=0, type=float,
                     help="Mean offset for internal noise (Default: 0)")
parser.add_argument("--noisestd", default=0.03, type=float,
                     help="Std of internal noise (Default: 0)")
parser.add_argument("--wandb_log", action='store_true', default=True,
                     help="Log training to Weights and Biases?")

args = parser.parse_args()

wandb_runner = wandb.init(
                        # set the wandb project where this run will be logged
                        entity = 'adel-halawa-mila', #'ahalawa-mcgill-university',
                        project = 'MultiEnv',
                        name=args.namext,
                        id = args.namext,
                        dir = args.netsfolder,
                        resume='allow',
                        )

# File management
savename = args.namext
figfolder = args.netsfolder + 'nets/'+args.savefolder+'trainfigs/'+savename

# Set seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

## Make environments and save dataset, if needed
envs = []
for envname in args.envlist:
    env = make_env(env_key=envname, package=args.env_package, act_enc=args.act_enc)
    agent = RandomActionAgent(env.action_space, np.array([0.15,0.15,0.6,0.1]))
    create_dataloader(env=env, agent=agent, n_trajs=args.datasetSize,
                      folder=args.datasetfolder, batch_size=args.batch_size, 
                      seq_length=args.seq_length, num_workers=0)
    envs.append(env)


## Make pRNN< add envs to library (note-can remove some options per this project)
predictiveNet = PredictiveNet(envs[0],
                                hidden_size=args.hiddensize,
                                pRNNtype=args.pRNNtype,
                                losstype=args.losstype,
                                learningRate = args.lr,
                                weight_decay = args.weight_decay,
                                trainNoiseMeanStd= (args.noisemean,args.noisestd),
                                trainBias = args.trainBias,
                                bias_lr = args.bias_lr,
                                identityInit = args.identityInit,
                                dataloader=True,
                                bptttrunc = args.bptttrunc,
                                neuralTimescale = args.ntimescale,
                                dropp=args.dropout,
                                wandb_log=args.wandb_log)
for env in envs[1:]:
    predictiveNet.addEnvironment(env)
    
predictiveNet.seed = args.seed
predictiveNet.trainArgs = args
predictiveNet.savefolder = args.savefolder
predictiveNet.savename = savename

#Set up sequence of envs (for epochs), based on the training curriculum
if args.trainCurriculum == 'sequential':
    envSequence = []
    envNames = []
    for env in envs:
        [envSequence.append(env) for i in range(args.epochsPerEnv)]
        [envNames.append(env.name) for i in range(args.epochsPerEnv)]
        
elif args.trainCurriculum == 'interleaved-epoch':
    envSequence = []
    envNames = []
    for i in range(args.epochsPerEnv):
        [envSequence.append(env) for env in envs]
        [envNames.append(env.name) for env in envs]
    
elif args.trainCurriculum == 'interleaved-batch':
    envMerged = mergeDatasets(envs, batch_size=args.batch_size, mixed_batch=False)
    envname = 'interleaved-batch'
    envSequence = []
    envNames = []
    [envSequence.append(envMerged) for i in range(args.epochsPerEnv*len(envs))]
    [envNames.append(envname) for i in range(args.epochsPerEnv*len(envs))]
    
elif args.trainCurriculum == 'mixed-batch':
    envMerged = mergeDatasets(envs, batch_size=args.batch_size, mixed_batch=True)
    envname = 'mixed-batch'
    envSequence = []
    envNames = []
    [envSequence.append(envMerged) for i in range(args.epochsPerEnv*len(envs))]
    [envNames.append(envname) for i in range(args.epochsPerEnv*len(envs))]
    
elif args.trainCurriculum == 'hold-first':
    envSequence = []
    envNames = []
    [envSequence.append(envs[0]) for i in range(args.epochsPerEnv*len(envs))]
    [envNames.append(envs[0].name) for i in range(args.epochsPerEnv*len(envs))]

elif args.trainCurriculum == 'hold-last':
    envSequence = []
    envNames = []
    [envSequence.append(envs[-1]) for i in range(args.epochsPerEnv*len(envs))]
    [envNames.append(envs[-1].name) for i in range(args.epochsPerEnv*len(envs))]


def calculateTrainingMetrics(predictiveNet):

    for enum,env in enumerate(envs):
        place_fields, SI, _ = predictiveNet.calculateSpatialRepresentation(env,agent,
                                                      trainDecoder=False,saveTrainingData=False,
                                                      bitsec= False,
                                                      calculatesRSA = True, sleepstd=0.03,
                                                      wandb_nameext='_env'+str(enum))
        
        sequence_duration = 2000
        obs,act,_,_ = predictiveNet.collectObservationSequence(env, 
                                                      agent, 
                                                      sequence_duration)
        obs_pred, obs_next, h = predictiveNet.predict(obs, act)
        predloss = predictiveNet.loss_fn(obs_pred, obs_next, h)
        wandb.log({'predloss_env'+str(enum): predloss.detach().numpy()})        
        #Metric for each environment...
            #- Mean SI (all, tuned cells)
            #- Sleep occupancy (? might need new way to calculate this, using all envs)

    numsleeps = 0
    MEGA = multiEnvGeometryAnalysis(predictiveNet,envs,
                                    timesteps_wake=8000,
                                    numsleeps=numsleeps, timesteps_sleep=200,
                                    withIsomap=True, withIsomap3d=True)
    fig = MEGA.IsomapOnlyFigure(show=False)
    wandb.log({"spatial_geometry": wandb.Image(fig)})
    plt.close(fig) 
    
    # predictiveNet.addTrainingData('sRSA_env0', MEGA.sRSA[0])
    # predictiveNet.addTrainingData('sRSA_env1', MEGA.sRSA[1])
    # predictiveNet.addTrainingData('D_01', MEGA.manifoldDistance[0])
    # predictiveNet.addTrainingData('D_10', MEGA.manifoldDistance[1])
    return


#Train 1 trial and calculatae initial metrics
numepochs = args.epochsPerEnv*len(envs)
sequence_duration = args.seq_length
num_trials = args.trials_per_epoch

predictiveNet.trainingCompleted = False
if predictiveNet.numTrainingTrials == -1:
    epoch = 0
    #Calculate initial spatial metrics etc
    print('Training Baseline')
    predictiveNet.trainingEpoch(envSequence[epoch], agent,
                            sequence_duration=sequence_duration,
                            num_trials=1)
    #predictiveNet.addTrainingData('envName', envNames[epoch])
    
    print('Calculating Performance Metrics')
    calculateTrainingMetrics(predictiveNet)

    print('Saving Network')
    predictiveNet.saveNet(args.savefolder+savename,savefolder = args.netsfolder)
    
    
#loop epochs
while predictiveNet.numTrainingEpochs<numepochs:
    epoch = predictiveNet.numTrainingEpochs
    print(f'Training Epoch {epoch}, Env: '+envNames[epoch])
    predictiveNet.trainingEpoch(envSequence[epoch], agent,
                            sequence_duration=sequence_duration,
                            num_trials=num_trials)
    #predictiveNet.addTrainingData('envName', envNames[epoch])
    
    print('Calculating Performance Metrics')
    calculateTrainingMetrics(predictiveNet)
    
    print('Saving Network')
    predictiveNet.saveNet(args.savefolder+savename,savefolder = args.netsfolder)
    #predictiveNet.saveNet(savefolder+savename)
    
predictiveNet.trainingCompleted = True    
    
    

# #If the user doesn't want to save all that training data, delete it except the last one
# if args.saveTrainData is False:
#     predictiveNet.TrainingSaver = predictiveNet.TrainingSaver.drop(predictiveNet.TrainingSaver.index[:-1])
#     predictiveNet.saveNet(args.savefolder+savename,savefolder = args.netsfolder)
