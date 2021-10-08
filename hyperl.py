from gym.utils import seeding
import os 
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['THEANO_FLAGS'] = f'mode=FAST_RUN,device=cpu,floatX=float32'
import numpy as np
import networkx as nx
from numpy import *
from numpy.random import *
import pandas as pd
import random as rnd
import keras
import sys
sys.path.append('/home/peter')
sys.path.append('/home/ubuntu')
#from universal import *
sys.path.append('/home/peter/code/projects')
sys.path.append('C:/Users/spook/Dropbox/code/projects')
sys.path.append('/home/peter/code/projects/deepneat')
sys.path.append('/home/ubuntu')
sys.path.append('/home/ubuntu/new/automl')
#from massimport import *
#from project_common import *
#from paramspace import *
#import project_common
import gym
from aidevutil import *
from keras.utils.vis_utils import plot_model
from tqdm import tqdm_notebook as tqdm
#from deepneat import *
from ipywidgets import interactive
from scipy.optimize import minimize
from sklearn.model_selection import *
from sklearn.preprocessing import OneHotEncoder
from dask import compute, delayed, persist
from dask.distributed import Client, wait
from dask.distributed import as_completed
import ipywidgets as widgets
import tensorflow as tf
from sklearn.metrics import *
from empyrical import sortino_ratio, calmar_ratio, omega_ratio
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, ActorCriticPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecEnv, VecEnvWrapper
from stable_baselines import A2C, PPO2, DQN, ACKTR, ACER#, TRPO
from stable_baselines.common.vec_env import DummyVecEnv
import matplotlib as mpl
from IPython.display import clear_output
import shutil
import time
import matplotlib.pyplot as plt
import gc
gc.enable()


def revsigm(data):
    return 1. - (1. / (1. + np.exp(-data)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def retanher(x):
    if x > 0:
        return np.tanh(x)
    else:
        return 0

retanh = np.vectorize(retanher)

def load_titanic():
    def Outputs(data):
        return np.round(revsigm(data))


    def MungeData(data):
        # Sex
        data.drop(['Ticket', 'Name'], inplace=True, axis=1)
        data.Sex.fillna('0', inplace=True)
        data.loc[data.Sex != 'male', 'Sex'] = 0
        data.loc[data.Sex == 'male', 'Sex'] = 1
        # Cabin
        cabin_const = 1
        data.Cabin.fillna(str(cabin_const), inplace=True)
        data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = cabin_const
        data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = cabin_const
        data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = cabin_const
        data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = cabin_const
        data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = cabin_const
        data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = cabin_const
        data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = cabin_const
        data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = cabin_const
        # Embarked
        data.loc[data.Embarked == 'C', 'Embarked'] = 1
        data.loc[data.Embarked == 'Q', 'Embarked'] = 2
        data.loc[data.Embarked == 'S', 'Embarked'] = 3
        data.Embarked.fillna(0, inplace=True)
        data.fillna(-1, inplace=True)
        return data.astype(float)


    train = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
    test = pd.read_csv("test.csv", dtype={"Age": np.float64}, )
    mdt = MungeData(train)

    bdy = mdt['Survived'].values
    bdx = mdt.drop(['Survived'], axis=1).values[:, 1:]
    bdx_test_df = MungeData(test)
    x_val_real = bdx_test_df.values[:, 1:]

    rtit = pd.read_csv('titanic_100p_score.csv')

    bdx_val = x_val_real
    bdy_val = rtit.values[:,1]

    return bdx, bdy, bdx_val, bdy_val

np.random.seed()
rnd.seed()

nrows = 32
nfeats = 8

bdx, bdy, bdx_val, bdy_val = load_titanic()

bdx = bdx[0:nrows]
bdy = bdy[0:nrows]
seed = 0

#from sklearn.datasets import make_classification
#bdx, bdy = make_classification(n_samples=nrows, n_features=nfeats)

#bdx_val = bdx
#bdy_val = bdy


num_inputs = bdx.shape[1]+1 #+ the bias
num_outputs = 1

max_nodes = num_inputs+num_outputs + 8
max_links = 8
prefer_incoming_links = 3
prefer_incoming_links_enabled = 0

max_weight = 2
max_init_weight = 2
# if true, the new links will have random weights, otherwise init_structural_with
random_init_structural = 1
init_structural_with = 1.0
# if true, some inputs can become disconnected
enable_disconnected_inputs = 1

# the last (bias) input
bias_const = 1.0
enforce_bias_link = 0


funcs = [#'add', #'mul', #'inv', 'neg', 'abs',
         #'div', 'arctan2', 'hypot', 'pow', 'sub', 'mod', 'hside', # all of these have arity 2
#      'sqrt', 'sqr',
#       'sig',
       'tanh',
#       'cbrt', 'cube',
#      'min', 'max', 'mean', 'span',
#      'std', 'var',
#      'argmin', 'argmax', 'sign',
#      'exp', 'log',
#      'sin', #'cos', #'tan', 'arcsin', 'arccos', 'arctan',
#      'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
        ]
len(funcs)

# the node matrix
# row - node idx
# col - func
nm = zeros((max_nodes-num_inputs, len(funcs)))
nm[:,0] = 1 # all are "add" initially

def num_nodes(m):
    sml = (np.sum(m[:,num_inputs:], axis=0) > 0).tolist()[::-1]
    if 1.0 not in sml:
        return num_inputs + num_outputs
    else:
        return (max_nodes - num_inputs - num_outputs -
                sml.index(1)) + num_inputs + num_outputs


def see_graph(m, pos=None, wm=None, nm=None):
    inpout_list = list(range(num_inputs)) + list(range(num_inputs + num_outputs))

    g = nx.DiGraph(m)
    # g.remove_nodes_from(nx.isolates(g))
    g.remove_nodes_from([x for x in list(nx.isolates(g)) if x not in inpout_list])
    if pos is None:
        pos = nx.spring_layout(g, k=50)

    if wm is not None:
        for i in range(max_nodes):
            for j in range(max_nodes):
                if m[i, j] == 1:
                    g[i][j]['weight'] = wm[i, j]

    nlabs = {}
    for i in range(max_nodes):
        if i in g.nodes():
            nlabs[i] = str(i)
    if nm is not None:
        for i in range(num_inputs, num_nodes(m)):
            if i in g.nodes():
                # g.node[i]['mm'] = funcs[argmax(nm[i-num_inputs,:].reshape(-1))]
                nlabs[i] = str(i) + '\n' + funcs[argmax(nm[i - num_inputs, :].reshape(-1))]

    cls = []
    if wm is not None:
        for i, o in g.edges():
            cls.append(g.adj[i][o]['weight'])

    nx.draw_networkx_nodes(g, pos=pos, node_size=800)
    nx.draw_networkx_labels(g, pos=pos, labels=nlabs, font_size=10);

    if cls:
        nx.drawing.nx_pylab.draw_networkx_edges(g, pos=pos,
                                                edge_vmin=-max_weight,
                                                edge_vmax=max_weight,
                                                edge_cmap=mpl.cm.get_cmap(name='jet'  # 'RdBu'
                                                                          ),
                                                width=1, edge_color=cls,
                                                alpha=1.0,
                                                arrows=1,
                                                node_size=800,
                                                arrowstyle='fancy',
                                                arrowsize=12
                                                );
    else:
        nx.drawing.nx_pylab.draw_networkx_edges(g, pos=pos,
                                                width=1,
                                                alpha=0.75,
                                                arrows=1,
                                                );


def is_ok(m, num_inputs, num_outputs, tell=0):
    inpout_list = list(range(num_inputs)) + list(range(num_inputs + num_outputs))

    if np.sum(m) == 0:
        if tell: print('no edges')
        return False

    g = nx.Graph(m)
    # g.remove_nodes_from(nx.isolates(g))
    g.remove_nodes_from([x for x in list(nx.isolates(g))
                         if x not in inpout_list])

    if enable_disconnected_inputs:
        trl = list(nx.connected_components(g))
        trl = [x for x in trl if not (len(x) == 1 and all([y in inpout_list for y in x]))]
        if len(trl) > 1:
            if tell: print('is not connected')
            return False
    else:
        if len(list(nx.connected_components(g))) > 1:
            if tell: print('is not connected')
            return False

    g = nx.DiGraph(m)
    g.remove_nodes_from(list(nx.isolates(g)))
    if not nx.dag.is_directed_acyclic_graph(g):
        if tell: print('is not DAG')
        return False

    # make sure the remaining nodes will be connected
    inps = list(range(num_inputs))
    outs = list(range(num_inputs, num_inputs + num_outputs))
    for k in g.nodes():
        if k in inps:
            # input
            if (len(list(g.out_edges(nbunch=k))) == 0):
                if tell: print('input', k, 'is not connected')
                return False
        elif k in outs:
            # output
            if (len(list(g.in_edges(nbunch=k))) == 0):
                if tell: print('output', k, 'is not connected')
                return False
        else:
            # hidden
            if (len(list(g.in_edges(nbunch=k))) == 0):
                if tell: print('node', k, 'has no incoming edges')
                return False

            if (len(list(g.out_edges(nbunch=k))) == 0):
                if tell: print('node', k, 'has no outgoing edges')
                return False

    return True


# add link
def add_link(m, si, di, num_inputs, num_outputs, max_links, wm=None):
    # outputs can't be sources
    if si in list(range(num_inputs, num_inputs + num_outputs)):
        return False
    # inputs can't be destinations
    if di in list(range(num_inputs)):
        return False

    if np.sum(m) >= max_links:
        return False

    tm = np.copy(m)
    tm[si, di] = 1

    if not is_ok(tm, num_inputs, num_outputs):
        # print("Not acyclic in add_link!")
        return False

    m[si, di] = 1

    if wm is not None:  # the weight matrix
        if random_init_structural:
            wm[si, di] = rnd.uniform(-max_init_weight, max_init_weight)
        else:
            wm[si, di] = init_structural_with

    return True


# remove link
def remove_link(m, si, di, num_inputs, num_outputs, wm=None):
    # don't allow removing links coming from bias
    if enforce_bias_link:
        if si == num_inputs - 1:
            return False

    tm = np.copy(m)
    tm[si, di] = 0

    if not is_ok(tm, num_inputs, num_outputs):
        # print("Not acyclic in remove_link!")
        return False

    m[si, di] = 0
    if wm is not None:  # the weight matrix
        wm[si, di] = 0
    return True


# add node
def add_node(m, si, di, num_inputs, num_outputs, max_links, wm=None, nm=None):
    if m[si, di] == 0:
        return False  # can't split nonexistent connections

    # don't split bias links
    if enforce_bias_link:
        if si == num_inputs - 1:
            return False

    if np.sum(m) >= max_links - 1:
        return False

    #     if (si, di) not in innovs.innovs:
    #         # new innovation
    #         inn = innovs.last_innov_num
    #         innovs.innovs[(si, di)] = inn
    #         innovs.last_innov_num += 1
    #     else:
    #         # known
    #         inn = innovs.innovs[(si, di)]

    if not any(np.sum(m[:, num_inputs + num_outputs:], axis=0) == 0):
        return False
    else:
        # there is a free slot, use that
        inn = argmin(np.sum(m[:, num_inputs + num_outputs:], axis=0)) + num_inputs + num_outputs

    m[si, di] = 0
    m[si, inn] = 1
    m[inn, di] = 1

    # hack
    # always add bias connection
    if enforce_bias_link:
        if si != num_inputs - 1:
            m[num_inputs - 1, inn] = 1

    if wm is not None:
        wm[si, di] = 0
        if random_init_structural:
            wm[si, inn] = rnd.uniform(-max_init_weight, max_init_weight)
            wm[inn, di] = rnd.uniform(-max_init_weight, max_init_weight)
            if enforce_bias_link:
                if si != num_inputs - 1:
                    wm[num_inputs - 1, inn] = rnd.uniform(-max_init_weight, max_init_weight)
        else:
            wm[si, inn] = init_structural_with
            wm[inn, di] = init_structural_with
            if enforce_bias_link:
                if si != num_inputs - 1:
                    wm[num_inputs - 1, inn] = init_structural_with

    if nm is not None:
        # random func
        if random_init_structural:
            ri = rnd.choice(arange(len(funcs)))
        else:
            ri = 0
        nm[inn - num_inputs, :] = 0
        nm[inn - num_inputs, ri] = 1

    return True


# remove a simple node (connected by 2 edges only)
def remove_node(m, ni, num_inputs, num_outputs, wm=None, nm=None):
    # also ensure only one incoming & outgoing edge to ni
    if (np.sum(m[:, ni]) != 1) or (np.sum(m[ni, :]) != 1):
        return False

    si = argmax(m[:, ni])
    di = argmax(m[ni, :])

    # can't remove inputs/outputs
    if ni < num_inputs + num_outputs:
        return False

    tm = np.copy(m)
    tm[si, ni] = 0
    tm[ni, di] = 0
    tm[si, di] = 1

    if not is_ok(tm, num_inputs, num_outputs):
        # print("Not acyclic in remove_node!")
        return False

    m[si, ni] = 0
    m[ni, di] = 0
    m[si, di] = 1

    if wm is not None:
        wm[si, ni] = 0
        wm[ni, di] = 0
        if wm[si, di] == 0:
            if random_init_structural:
                wm[si, di] = rnd.uniform(-max_init_weight, max_init_weight)
            else:
                wm[si, di] = init_structural_with

    if nm is not None:
        nm[ni - num_inputs, :] = 0

    return True

action_codes = [
"Move link caret up",
"Move link caret down",
"Move link caret left",
"Move link caret right",
#"Randomize link caret position",

"Move node caret up",
"Move node caret down",
#"Randomize node caret",

"Attempt to add link at caret position",
"Attempt to remove link at caret position",
"Attempt to add node at caret position",
"Attempt to remove node at caret position",

"Mutate weight at link caret position +",
"Mutate weight at link caret position -",
#"Randomize weight at link caret position",
#"Set weight to 1.0 at link caret position",

"Mutate node at node caret position +",
"Mutate node at node caret position -",
#"Randomize node at node caret position",
#"Set node to default at node caret position"
#"Request testing"
]

len(action_codes)


def make_input(m, wm, nm, caret_row, caret_col, node_caret,
               max_nodes, max_links,
               num_inputs, num_outputs, nnoutput,
               la=0, ls=0, nnc=0, nlc=0):
    iii = []
    # The adjacency matrix's operating field
    iii += [m[:, num_inputs:].reshape(-1)]
    # The weight matrix
    iii += [wm[:, num_inputs:].reshape(-1) / max_weight]
    # The link caret x/y position
    i = np.zeros(max_nodes)
    i[caret_row] = 1
    iii += [i]
    i = np.zeros(max_nodes)
    i[caret_col] = 1
    iii += [i]
    # What's under the link caret position (link or not)
    iii += [array([m[caret_row, caret_col]])]
    # What weight is under the caret position (0 if no link)
    iii += [array([wm[caret_row, caret_col]])]
    # The node matrix
    iii += [nm.reshape(-1)]
    # The node caret position
    i = np.zeros(max_nodes - num_inputs)
    i[node_caret] = 1
    iii += [i]
    # What's under the node caret
    iii += [nm[node_caret, :].reshape(-1)]
    # Amount of links / limit
    iii += [array([np.sum(m) / max_links])]
    # Amount of nodes / limit
    iii += [array([num_nodes(m) / max_nodes])]
    # Sum of link caret row / num_nodes
    iii += [array([np.sum(m[caret_row, :]) / num_nodes(m)])]
    # Sum of link caret col / num_nodes
    iii += [array([np.sum(m[:, caret_col]) / num_nodes(m)])]
    # Link caret is at diagonal or not
    if caret_row == caret_col:
        iii += [array([1])]
    else:
        iii += [array([0])]
    # Node removal is possible at link caret position or not
    if (caret_row == caret_col) and ((sum(m[:, caret_col]) == 1) and (sum(m[caret_row, :]) == 1)):
        iii += [array([1])]
    else:
        iii += [array([0])]

    # Last action was successful or not
    iii += [array([la])]

    # Last score
    iii += [array([ls])]

    # charge for nodes
    iii += [array([nnc])]

    # charge for links
    iii += [array([nlc])]

    # the NN's output over the data
    #iii += [nnoutput]

    inp = hstack(iii)
    return inp

"""
exmpi = make_input(m, m, nm, 0, 2, 0, max_nodes, max_links, num_inputs, num_outputs)
exmpi.shape
"""

def activate_graph(gr, inputs, num_outputs=1):
    num_inputs = inputs.shape[0]
    allnodes = list(nx.dfs_postorder_nodes(gr))[::-1]
    for a in allnodes: gr.node[a]['act'] = None

    # separate input from non-input nodes
    allnodes = [x for x in allnodes if x > (num_inputs - 1)]

    # input the data
    for i, inp in zip(range(0, num_inputs), inputs):
        gr.node[i]['act'] = inp

    # pass through the graph
    for an in allnodes:
        # collect the inputs to this node
        mm = gr.node[an]['mm']

        # also sort the incoming edges by id for consistency
        inedg = list(gr.in_edges(nbunch=an))

        # inedg = sorted(inedg, key = lambda x: x[0])

        inps = [gr.node[i]['act'] for i, o in inedg]
        inedgw = list(gr.in_edges(nbunch=an, data=True))

        # print(inedgw)
        ws = [ts['weight'] for i, o, ts in inedgw]
        # weighted stack
        inps = np.vstack([w * x for w, x in zip(ws, inps)])

        sact = np.sum(inps, axis=0)  # this node's default activation
        act = sact

        try:
            if mm == 'add':
                act = sact
            if mm == 'neg':
                act = -sact
            if mm == 'mul':
                act = np.prod(inps, axis=0)
            if mm == 'inv':
                act = 1.0 / sact
            if mm == 'sqr':
                act = sact ** 2
            if mm == 'cube':
                act = sact ** 3
            if mm == 'sqrt':
                act = np.sqrt(sact)
            if mm == 'sig' or mm == 'sigmoid':
                act = sigmoid(sact)
            if mm == 'cbrt':
                act = np.cbrt(sact)
            if mm == 'sin':
                act = np.sin(sact)
            if mm == 'cos':
                act = np.cos(sact)
            if mm == 'tan':
                act = np.tan(sact)
            if mm == 'arcsin':
                act = np.arcsin(sact)
            if mm == 'arccos':
                act = np.arccos(sact)
            if mm == 'arctan':
                act = np.arctan(sact)
            if mm == 'log':
                act = np.log(sact)
            if mm == 'exp':
                act = np.exp(sact)
            if mm == 'abs':
                act = np.abs(sact)
            if mm == 'sinh':
                act = np.sinh(sact)
            if mm == 'cosh':
                act = np.cosh(sact)
            if mm == 'tanh':
                act = np.tanh(sact)
            if mm == 'arcsinh':
                act = np.arcsinh(sact)
            if mm == 'arccosh':
                act = np.arccosh(sact)
            if mm == 'arctanh':
                act = np.arctanh(sact)
            if mm == 'min':
                act = np.min(inps, axis=0)
            if mm == 'max':
                act = np.max(inps, axis=0)
            if mm == 'mean':
                act = np.mean(inps, axis=0)
            if mm == 'span':
                act = np.max(inps, axis=0) - np.min(inps, axis=0)
            if mm == 'var':
                act = np.var(inps, axis=0)
            if mm == 'std':
                act = np.std(inps, axis=0)
            if mm == 'argmax':
                act = np.argmax(inps, axis=0) / inps.shape[0]  # normalized argmax
            if mm == 'argmin':
                act = np.argmin(inps, axis=0) / inps.shape[0]  # normalized argmax
            if mm == 'sign':
                act = np.sign(sact)
            # arity 2
            if mm == 'div':
                if inps.shape[0] > 1:
                    act = inps[0] / inps[1]
                else:
                    act = inps[0]
            if mm == 'arctan2':
                if inps.shape[0] > 1:
                    act = np.arctan2(inps[0], inps[1])
                else:
                    act = inps[0]
            if mm == 'hypot':
                if inps.shape[0] > 1:
                    act = np.hypot(inps[0], inps[1])
                else:
                    act = inps[0]
            if mm == 'pow':
                if inps.shape[0] > 1:
                    act = np.power(inps[0], inps[1])
                else:
                    act = inps[0]
            if mm == 'sub':
                if inps.shape[0] > 1:
                    act = np.subtract(inps[0], inps[1])
                else:
                    act = inps[0]
            if mm == 'mod':
                if inps.shape[0] > 1:
                    act = np.mod(inps[0], inps[1])
                else:
                    act = inps[0]
            if mm == 'hside':
                if inps.shape[0] > 1:
                    act = np.heaviside(inps[0], inps[1])
                else:
                    act = inps[0]

        except Exception as ex:
            print(ex)
            act = sact

        # store activation
        gr.node[an]['act'] = act

        # print('act:', act)
    outputs = [gr.node[o]['act'] for o in allnodes[-num_outputs:]]
    return outputs


score_metric = accuracy_score


def test_classification(m, wm, nm, num_inputs, num_outputs, max_nodes):
    inpout_list = list(range(num_inputs)) + list(range(num_inputs + num_outputs))
    g = nx.DiGraph(m)
    g.remove_nodes_from([x for x in list(nx.isolates(g))
                         if x not in inpout_list])

    for i in range(max_nodes):
        for j in range(max_nodes):
            if m[i, j] == 1:
                g.edges[i, j]['weight'] = wm[i, j]

    for i in range(num_inputs, num_nodes(m)):
        if i in g.nodes():
            g.node[i]['mm'] = funcs[np.argmax(nm[i - num_inputs, :].reshape(-1))]

    p1 = activate_graph(g, vstack([bdx.T,
                                   ones(bdx.shape[0]) * bias_const]),
                       num_outputs=num_outputs)[0]

    t = ((np.tanh(p1) + 1.0) / 2)
    t[isnan(t)] = 0
    t[isinf(t)] = 0
    acc = score_metric(bdy, np.round(t))
    acc_real = acc

    p2 = activate_graph(g, vstack([bdx_val.T, ones(bdx_val.shape[0]) * bias_const]),
                       num_outputs=num_outputs)[0]
    t1 = ((np.tanh(p2) + 1.0) / 2)
    t1[isnan(t1)] = 0
    t1[isinf(t1)] = 0
    acc1 = score_metric(bdy_val, np.round(t1))
    acc1_real = acc1

    return float(acc), float(acc1), float(acc_real), float(acc1_real), t



def make_pos(m, wm):
    inpout_list = list(range(num_inputs)) + list(range(num_inputs + num_outputs))
    g = nx.DiGraph(m)

    for i in range(max_nodes):
        for j in range(max_nodes):
            if m[i, j] != 0:
                g.edges[i, j]['weight'] = wm[i, j]

    pos = {}
    # will provide my own coords
    for i, x in zip(range(num_inputs), linspace(0, 1, num_inputs)):
        pos[i] = array([x, 1])

    if num_outputs == 1:
        pos[num_inputs] = array([0.5, 0])
    elif num_outputs == 2:
        pos[num_inputs] = array([0.25, 0])
        pos[num_inputs + 1] = array([0.75, 0])
    else:
        for i, x in zip(range(num_inputs, num_inputs + num_outputs),
                        linspace(0, 1, num_outputs)):
            pos[i] = array([x, 0])

    for i, x in zip(range(num_inputs + num_outputs, max_nodes),
                    linspace(0, 1, max_nodes - (num_inputs + num_outputs))):
        if i % 2 == 0:
            ofs = 0.07
        else:
            ofs = -0.15
        pos[i] = array([x, 0.5 + ofs])
    return pos



env_type = 'discrete' # or discrete

add_node_prob = 1.0
add_link_prob = 1.0
rem_node_prob = 1.0
rem_link_prob = 1.0

mut_weight_prob = 1.0
replace_weight_prob = 1.0
defaultize_weight_prob = 1.0

min_mut_power = 0.01
max_mut_power = 0.1

min_extra_links = 2
min_extra_links_enabled = 0

max_steps = 10000
max_stag = 10000

steps_to_charge_newnode = 100
steps_to_charge_newlink = 10

display_every = 1000
display_matrix = 0
display_weight_matrix = 0
display_sleep = 0
display_graphs = 1

penalize_unsuccessful_move = 0
penalize_unsuccessful_move_mul = 0.9

penalize_unsuccessful_modify = 0
penalize_unsuccessful_modify_mul = 0.5

penalize_unsuccessful_struct = 0
penalize_unsuccessful_struct_mul = 0.5

# caret movement will wrap around, it can never be unsuccessful
infinite_caret = 1

# implemented are:
# sparse_diff_best - given when it beats best record, 0 otherwise
# sparse_diff_prev - given when it beats previous score, 0 otherwise
# dense_abs - always given the current score
# dense_diff_best - always given the (current score - best)
# dense_diff_prev - always given the (current score - previous)
# dense_diff_mean_prev - always given the (current score - mean of previous N scores)
reward_type = 'dense_abs'

last_scores_window = 30 # parameter for dense_diff_mean_prev


class GraphMakerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        return self.step(action)

    def _reset(self):
        return self.reset()

    def _render(self, mode='human', close=False):
        # ... TODO
        pass

def __init__(self, max_nodes, max_links, num_inputs, num_outputs):

    self.max_nodes = max_nodes
    self.max_links = max_links
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs

    m = zeros((max_nodes, max_nodes))
    m[0:num_inputs, num_inputs:num_inputs + num_outputs] = 1
    m[3, 8] = 0
    wm = randn(max_nodes, max_nodes)
    self.ppos = make_pos(m, wm)

    if env_type == 'box':
        self.action_space = gym.spaces.Box(-1*np.ones(5), np.ones(5))
    else:
        self.action_space = gym.spaces.Discrete(len(action_codes))

    self.matrix = zeros((self.max_nodes, self.max_nodes))
    self.weight_matrix = zeros((self.max_nodes, self.max_nodes))
    self.node_matrix = zeros((self.max_nodes-self.num_inputs, len(funcs)))
#     self.matrix[0:self.num_inputs,
#                 self.num_inputs:self.num_inputs+self.num_outputs] = 1
#     for i in range(self.max_nodes):
#         for j in range(self.max_nodes):
#             if self.matrix[i,j]==1:
#                 self.weight_matrix[i,j] = 1.0
#     for i in range(self.num_inputs, num_nodes(self.matrix)):
#         self.node_matrix[i-self.num_inputs, rnd.choice(arange(len(funcs)))]=1

    self.caret_row = 0
    self.caret_col = self.num_inputs
    self.node_caret = 0

    #f, fr, facc, faccr, nnout = test_xor(self.matrix, self.weight_matrix, self.node_matrix,
    #                                     self.num_inputs, self.num_outputs, self.max_nodes)

    exmpi = make_input(self.matrix, self.weight_matrix, self.node_matrix,
                              self.caret_row, self.caret_col, self.node_caret,
                              self.max_nodes, self.max_links,
                              self.num_inputs, self.num_outputs, zeros(bdx.shape[0]))

    self.observation_space = gym.spaces.Box(np.zeros(exmpi.shape[0]),
                                        np.ones(exmpi.shape[0]), )
    self.best_ever = (0, 0)
    self.prev_score = [(0, 0, 0, 0)]
    self.reset()

GraphMakerEnv.__init__ = __init__


def move(self, action):
    move_attempt = 0
    move_success = 0
    modify_attempt = 0
    modify_success = 0
    struct_attempt = 0
    struct_success = 0

    wants_test = False

    # execute action
    if action == 0:
        # move caret up
        if not infinite_caret:
            move_attempt = 1
            if self.caret_row > 0:
                self.caret_row -= 1
                move_success = 1
            else:
                move_success = 0
        else:
            move_attempt = 1
            move_success = 1
            self.caret_row -= 1
            if self.caret_row <= -1:
                self.caret_row = num_nodes(self.matrix) - 1


    elif action == 1:
        # move caret down
        if not infinite_caret:
            move_attempt = 1
            if self.caret_row < num_nodes(self.matrix) - 1:
                self.caret_row += 1
                move_success = 1
            else:
                move_success = 0
        else:
            move_attempt = 1
            move_success = 1
            self.caret_row += 1
            if self.caret_row >= num_nodes(self.matrix):
                self.caret_row = 0

    elif action == 2:
        # move caret left
        if not infinite_caret:
            move_attempt = 1
            if self.caret_col > self.num_inputs:
                self.caret_col -= 1
                move_success = 1
            else:
                move_success = 0
        else:
            move_attempt = 1
            move_success = 1
            self.caret_col -= 1
            if self.caret_col <= self.num_inputs - 1:
                self.caret_col = num_nodes(self.matrix) - 1

    elif action == 3:
        # move caret right
        if not infinite_caret:
            move_attempt = 1
            if self.caret_col < num_nodes(self.matrix) - 1:
                self.caret_col += 1
                move_success = 1
            else:
                move_success = 0
        else:
            move_attempt = 1
            move_success = 1
            self.caret_col += 1
            if self.caret_col >= num_nodes(self.matrix):
                self.caret_col = self.num_inputs

    #     elif action == 4:
    #         # randomize caret
    #         move_attempt = 1
    #         prv = (self.caret_row, self.caret_col)
    #         self.caret_row = rnd.randint(0, num_nodes(self.matrix)-1)
    #         self.caret_col = rnd.randint(self.num_inputs, num_nodes(self.matrix)-1)
    #         if (self.caret_row, self.caret_col) != prv:
    #             move_success = 1
    #         else:
    #             move_success = 0

    elif action == 4:
        # move node caret up
        if not infinite_caret:
            move_attempt = 1
            if self.node_caret > 0:
                self.node_caret -= 1
                move_success = 1
            else:
                move_success = 0
        else:
            move_attempt = 1
            move_success = 1
            self.node_caret -= 1
            if self.node_caret <= -1:
                self.node_caret = num_nodes(self.matrix) - self.num_inputs - 1

    elif action == 5:
        # move node caret down
        if not infinite_caret:
            move_attempt = 1
            if self.node_caret < num_nodes(self.matrix) - self.num_inputs - 1:
                self.node_caret += 1
                move_success = 1
            else:
                move_success = 0
        else:
            move_attempt = 1
            move_success = 1
            self.node_caret += 1
            if self.node_caret >= num_nodes(self.matrix) - self.num_inputs:
                self.node_caret = 0

    #     elif action == 7:
    #         # randomize node caret
    #         move_attempt = 1
    #         prv = self.node_caret
    #         self.node_caret = rnd.randint(0, num_nodes(self.matrix)-self.num_inputs-1)
    #         if self.node_caret != prv:
    #             move_success = 1
    #         else:
    #             move_success = 0

    elif action == 6:
        # add link attempt
        struct_attempt = 1
        if (rnd.uniform(0, 1) < add_link_prob) and (self.newlink_charge <= 0) and add_link(self.matrix, self.caret_row,
                                                                                           self.caret_col,
                                                                                           self.num_inputs,
                                                                                           self.num_outputs,
                                                                                           self.max_links,
                                                                                           wm=self.weight_matrix):
            struct_success = 1
            self.newlink_charge = steps_to_charge_newlink
        else:
            struct_success = 0

    elif action == 7:
        # remove link attempt
        struct_attempt = 1
        if (rnd.uniform(0, 1) < rem_link_prob) and remove_link(self.matrix, self.caret_row, self.caret_col,
                                                               self.num_inputs, self.num_outputs,
                                                               wm=self.weight_matrix):
            struct_success = 1
            # also boost charge by 50%
            self.newlink_charge *= 0.5
        else:
            struct_success = 0

    elif action == 8:
        # add node attempt
        struct_attempt = 1
        if (rnd.uniform(0, 1) < add_node_prob) and (self.newnode_charge <= 0) and add_node(self.matrix, self.caret_row,
                                                                                           self.caret_col,
                                                                                           self.num_inputs,
                                                                                           self.num_outputs,
                                                                                           self.max_links,
                                                                                           wm=self.weight_matrix,
                                                                                           nm=self.node_matrix):
            struct_success = 1
            self.newnode_charge = steps_to_charge_newnode
        else:
            struct_success = 0

    elif action == 9:
        # remove node attempt
        struct_attempt = 1
        if (rnd.uniform(0, 1) < rem_link_prob) and remove_node(self.matrix, self.caret_row,
                                                               self.num_inputs, self.num_outputs,
                                                               wm=self.weight_matrix,
                                                               nm=self.node_matrix):
            struct_success = 1
            # also boost charge by 50%
            self.newnode_charge *= 0.5
        else:
            struct_success = 0

    elif action == 10:
        # mutate parameter +
        modify_attempt = 1
        if (rnd.uniform(0, 1) < mut_weight_prob) and (self.matrix[self.caret_row,
                                                                  self.caret_col] == 1):
            self.weight_matrix[self.caret_row, self.caret_col] += \
                rnd.uniform(min_mut_power, max_mut_power)
            modify_success = 1
        else:
            modify_success = 0

    elif action == 11:
        # mutate parameter -
        modify_attempt = 1
        if (rnd.uniform(0, 1) < mut_weight_prob) and (self.matrix[self.caret_row,
                                                                  self.caret_col] == 1):
            self.weight_matrix[self.caret_row, self.caret_col] -= \
                rnd.uniform(min_mut_power, max_mut_power)
            modify_success = 1
        else:
            modify_success = 0

    #     elif action == 14:
    #         # randomize parameter
    #         modify_attempt = 1
    #         if (rnd.uniform(0,1)<replace_weight_prob) and (self.matrix[self.caret_row,
    #                                                                    self.caret_col] == 1):
    #             self.weight_matrix[self.caret_row, self.caret_col] = \
    #                             rnd.uniform(-max_init_weight, max_init_weight)
    #             modify_success = 1
    #         else:
    #             modify_success = 0

    #     elif action == 12:
    #         # set parameter to 1
    #         modify_attempt = 1
    #         if (rnd.uniform(0,1)<defaultize_weight_prob) and (self.matrix[self.caret_row,
    #                                                                       self.caret_col] == 1):
    #             self.weight_matrix[self.caret_row, self.caret_col] = init_structural_with
    #             modify_success = 1
    #         else:
    #             modify_success = 0

    elif action == 12:
        # mutate node parameter +
        modify_attempt = 1
        if np.sum(self.node_matrix[self.node_caret, :]) == 1:
            ocp = argmax(self.node_matrix[self.node_caret, :])
            ncp = ocp + 1
            cp = np.clip(ncp, 0, len(funcs) - 1)
            self.node_matrix[self.node_caret, :] = 0
            self.node_matrix[self.node_caret, cp] = 1
            if cp != ocp:
                modify_success = 1
            else:
                modify_success = 0
        else:
            modify_success = 0

    elif action == 13:
        # mutate node parameter -
        modify_attempt = 1
        if np.sum(self.node_matrix[self.node_caret, :]) == 1:
            ocp = argmax(self.node_matrix[self.node_caret, :])
            ncp = ocp - 1
            cp = np.clip(ncp, 0, len(funcs) - 1)
            self.node_matrix[self.node_caret, :] = 0
            self.node_matrix[self.node_caret, cp] = 1
            if cp != ocp:
                modify_success = 1
            else:
                modify_success = 0
        else:
            modify_success = 0

    #     elif action == 18:
    #         # randomize node parameter
    #         modify_attempt = 1
    #         if np.sum(self.node_matrix[self.node_caret, :])==1:
    #             ocp = argmax(self.node_matrix[self.node_caret, :])
    #             ncp = rnd.choice(arange(len(funcs)))
    #             self.node_matrix[self.node_caret, :] = 0
    #             self.node_matrix[self.node_caret, ncp] = 1
    #             if ocp != ncp:
    #                 modify_success = 1
    #             else:
    #                 modify_success = 0
    #         else:
    #             modify_success = 0

    #     elif action == 15:
    #         # set node parameter to default
    #         modify_attempt = 1
    #         if np.sum(self.node_matrix[self.node_caret, :])==1:
    #             ocp = argmax(self.node_matrix[self.node_caret, :])
    #             ncp = 0
    #             self.node_matrix[self.node_caret, :] = 0
    #             self.node_matrix[self.node_caret, ncp] = 1
    #             if ocp != ncp:
    #                 modify_success = 1
    #             else:
    #                 modify_success = 0
    #         else:
    #             modify_success = 0

    #         elif action == 14:
    #             # save state
    #             self.saved_matrix = np.copy(self.matrix)
    #             self.saved_weight_matrix = np.copy(self.weight_matrix)
    #             self.saved_caret = (self.caret_row, self.caret_col)
    #             success = 1
    #             modified = False

    #         elif action == 15:
    #             # restore state
    #             self.matrix = np.copy(self.saved_matrix)
    #             self.weight_matrix = np.copy(self.saved_weight_matrix)
    #             self.caret_row, self.caret_col = self.saved_caret
    #             success = 1
    #             modified = True

    #     elif action == 14:
    #         # request testing
    #         wants_test = True
    return move_attempt, move_success, modify_attempt, modify_success, \
           struct_attempt, struct_success, wants_test


GraphMakerEnv.move = move


def move_box(self, actions):
    # interpret all actions from the start
    caret_speed_row = 0
    if actions[0] < -0.33: caret_speed_row = -1
    if actions[0] > 0.33: caret_speed_row = 1

    caret_speed_col = 0
    if actions[1] < -0.33: caret_speed_col = -1
    if actions[1] > 0.33: caret_speed_col = 1

    node_caret_speed = 0
    if actions[2] < -0.33: node_caret_speed = -1
    if actions[2] > 0.33: node_caret_speed = 1

    #     link_decision = 0
    #     if actions[3] < -0.33: link_decision = -1
    #     if actions[3] > 0.33: link_decision = 1

    #     node_decision = 0
    #     if actions[4] < -0.33: node_decision = -1
    #     if actions[4] > 0.33: node_decision = 1

    new_weight = actions[3] * max_weight
    new_func = clip(int(((actions[4] + 1) / 2) * (len(funcs) - 1)),
                    0, len(funcs) - 1)

    # execute actions

    # move carets
    self.caret_row += caret_speed_row
    self.caret_col += caret_speed_col
    self.node_caret += node_caret_speed

    if not infinite_caret:
        self.caret_row = clip(self.caret_row, 0, num_nodes(self.matrix) - 1)
        self.caret_col = clip(self.caret_col,
                              self.num_inputs, num_nodes(self.matrix) - 1)
        self.node_caret = clip(self.node_caret, 0, num_nodes(self.matrix) - 1)
    else:
        if self.caret_row < 0: self.caret_row = (num_nodes(self.matrix) - 1)
        if self.caret_row > (num_nodes(self.matrix) - 1): self.caret_row = 0
        if self.caret_col < self.num_inputs: self.caret_col = (num_nodes(self.matrix) - 1)
        if self.caret_col > (num_nodes(self.matrix) - 1): self.caret_col = self.num_inputs
        if self.node_caret < 0: self.node_caret = num_nodes(self.matrix)
        if self.node_caret > num_nodes(self.matrix): self.node_caret = 0

    # execute link decision
    #     if link_decision == -1:
    #         if (rnd.uniform(0,1)<rem_link_prob) and \
    #             remove_link(self.matrix, self.caret_row, self.caret_col,
    #                         self.num_inputs, self.num_outputs,
    #                         wm = self.weight_matrix):
    #                     # also boost charge by 50%
    #                     self.newlink_charge *= 0.5
    #     elif link_decision == 1:
    #         if (rnd.uniform(0,1)<add_link_prob) and \
    #             (self.newlink_charge <= 0) and \
    #             add_link(self.matrix, self.caret_row, self.caret_col,
    #                     self.num_inputs, self.num_outputs, self.max_links,
    #                     wm = self.weight_matrix):
    #             self.newlink_charge = steps_to_charge_newlink

    #     # execute node decision
    #     if node_decision == -1:
    #         if (rnd.uniform(0,1)<rem_link_prob) and \
    #             remove_node(self.matrix, self.caret_row,
    #                         self.num_inputs, self.num_outputs,
    #                         wm = self.weight_matrix,
    #                         nm = self.node_matrix):
    #             # also boost charge by 50%
    #             self.newnode_charge *= 0.5
    #     elif node_decision == 1:
    #         if (rnd.uniform(0,1)<add_node_prob) and (self.newnode_charge <= 0) and \
    #             add_node(self.matrix, self.caret_row, self.caret_col,
    #                      self.num_inputs, self.num_outputs, self.max_links,
    #                      wm = self.weight_matrix,
    #                      nm = self.node_matrix):
    #                 self.newnode_charge = steps_to_charge_newnode

    # set the new parameters is possible
    # new weight
    if (rnd.uniform(0, 1) < mut_weight_prob) and \
            (self.matrix[self.caret_row,
                         self.caret_col] == 1):
        self.weight_matrix[self.caret_row, self.caret_col] = new_weight
    # new func
    #     try:
    #         if np.sum(self.node_matrix[self.node_caret, :])==1:
    #             self.node_matrix[self.node_caret, :] = 0
    #             self.node_matrix[self.node_caret, new_func] = 1
    #     except IndexError:
    #         pass

    self.caret_row = int(self.caret_row)
    self.caret_col = int(self.caret_col)
    self.node_caret = int(self.node_caret)


GraphMakerEnv.move_box = move_box


def step(self, action):
    self.step_counter += 1

    # for testing
    # action = rnd.choice(arange(15))
    if env_type == 'box':
        move_attempt, move_success, modify_attempt, modify_success, \
        struct_attempt, struct_success, wants_test = 1, 1, 1, 1, 1, 1, 1
        self.move_box(action)
    else:
        move_attempt, move_success, modify_attempt, modify_success, \
        struct_attempt, struct_success, wants_test = self.move(action)

    self.newnode_charge -= 1
    self.newlink_charge -= 1

    self.caret_row = clip(self.caret_row, 0, num_nodes(self.matrix) - 1)
    self.caret_col = clip(self.caret_col, self.num_inputs, num_nodes(self.matrix) - 1)
    self.node_caret = clip(self.node_caret, 0, num_nodes(self.matrix) - num_inputs - 1)
    self.weight_matrix = clip(self.weight_matrix, -max_weight, max_weight)

    self.newnode_charge = clip(self.newnode_charge, 0, steps_to_charge_newnode)
    self.newlink_charge = clip(self.newlink_charge, 0, steps_to_charge_newlink)

    info = {}
    done = False

    # reevaluate if modified
    if struct_success or modify_success:
        f, fr, facc, faccr, nnout = test_classification(self.matrix, self.weight_matrix, self.node_matrix,
                                                        self.num_inputs, self.num_outputs, self.max_nodes)
    else:
        f, fr, facc, faccr, nnout = self.prev_score[-1]

    reward = 0
    if reward_type == 'sparse_diff_best':
        if (f - self.best[0]) > 0.0:
            reward = (f - self.best[0]) * f
        else:
            reward = 0
    elif reward_type == 'sparse_diff_prev':
        if (f - self.prev_score[-1][0]) > 0.0:
            reward = (f - self.prev_score[-1][0]) * f
        else:
            reward = 0

    elif reward_type == 'dense_abs':
        reward = f
    elif reward_type == 'dense_diff_best':
        reward = f - self.best[0]
    elif reward_type == 'dense_diff_prev':
        reward = f - self.prev_score[-1][0]
    elif reward_type == 'dense_diff_mean_prev':
        scs = mean([x[0] for x in self.prev_score[-last_scores_window:]])
        reward = f - scs

    # Soft constraints applied to the reward
    if prefer_incoming_links_enabled:
        g = nx.DiGraph(self.matrix)
        g.remove_nodes_from(list(nx.isolates(g)))
        # make sure the remaining nodes will be connected
        inps = list(range(self.num_inputs))
        # outs=list(range(self.num_inputs, self.num_inputs+self.num_outputs))
        for k in g.nodes():
            if (k not in inps):  # and (k not in outs):
                # must be hidden
                if (len(list(g.in_edges(nbunch=k))) != prefer_incoming_links):
                    reward *= 0.2

    if min_extra_links_enabled:
        if enable_disconnected_inputs:
            if np.sum(self.matrix) < 1 + min_extra_links:
                reward = 0  # force at least one hidden node
                f, fr = 0, 0
        else:
            if np.sum(self.matrix) < self.num_inputs + min_extra_links:
                reward = 0  # force at least one hidden node
                f, fr = 0, 0

    if penalize_unsuccessful_move:
        if move_attempt and (not move_success):
            reward *= penalize_unsuccessful_move_mul
    if penalize_unsuccessful_modify:
        if modify_attempt and (not modify_success):
            reward *= penalize_unsuccessful_modify_mul
    if penalize_unsuccessful_struct:
        if struct_attempt and (not struct_success):
            reward *= penalize_unsuccessful_struct_mul


    if (f > self.best[0]) or (self.step_counter % display_every == 0):
        clear_output(wait=True)
        if display_graphs:# and (f > self.best_ever[0]):
            see_graph(self.matrix, pos=self.ppos,
                      wm=self.weight_matrix, nm=self.node_matrix)
            plt.show()
        print("%d steps, %3.4f / %3.4f, best: %3.4f / %3.4f\nbest ever: %3.4f / %3.4f" %
              (self.step_counter, f, fr,
               self.best[0], self.best[1],
               self.best_ever[0], self.best_ever[1]))
        print('Accuracy scores: %3.4f / %3.4f' % (facc, faccr))
        if env_type != 'box':
            print('Action: %s\nReward: %3.5f, charge for n/l: %3.4f/%3.4f'
                  % (action_codes[action], reward, 1.0 - self.newnode_charge / steps_to_charge_newnode,
                     1.0 - self.newlink_charge / steps_to_charge_newlink))
        else:
            print('Reward: %3.5f, charge for n/l: %3.4f/%3.4f'
                  % (reward, 1.0 - self.newnode_charge / steps_to_charge_newnode,
                     1.0 - self.newlink_charge / steps_to_charge_newlink))
        print('Caret row/col: %d/%d' % (self.caret_row, self.caret_col))

        if display_matrix:
            sm = np.copy(self.matrix)
            sm[self.caret_row, self.caret_col] = 666
            print(sm[0:num_nodes(self.matrix),
                  self.num_inputs:num_nodes(self.matrix)])
            # print(sm)
        if display_weight_matrix:
            print(self.weight_matrix[0:num_nodes(self.matrix),
                  self.num_inputs:num_nodes(self.matrix)])
            # print(self.weight_matrix)

        if display_sleep > 0:
            time.sleep(display_sleep)

    if f > self.best[0]:
        self.best = f, fr
        self.stag = 0
    else:
        self.stag += 1

    if f > self.best_ever[0]:
        self.best_ever = f, fr

    if (self.step_counter > max_steps) or (self.stag > max_stag):
        done = True

    observation = make_input(self.matrix, self.weight_matrix, self.node_matrix,
                             self.caret_row, self.caret_col, self.node_caret,
                             self.max_nodes, self.max_links,
                             self.num_inputs, self.num_outputs, nnout,
                             la=float(move_success or modify_success or struct_success),
                             ls=f,
                             nnc=1.0 - self.newnode_charge / steps_to_charge_newnode,
                             nlc=1.0 - self.newlink_charge / steps_to_charge_newlink, )

    self.prev_score.append((f, fr, facc, faccr, nnout))
    self.prev_reward.append(reward)

    self.prev_score = self.prev_score[-50:]
    self.prev_reward = self.prev_reward[-50:]

    return observation, reward, done, info


GraphMakerEnv.step = step


def reset(self):
    print('Env reset')
    self.step_counter = 0
    self.stag = 0

    self.matrix = zeros((self.max_nodes, self.max_nodes))
    self.weight_matrix = zeros((self.max_nodes, self.max_nodes))
    self.node_matrix = zeros((self.max_nodes - self.num_inputs, len(funcs)))
    if enable_disconnected_inputs:
        cn = rnd.randint(0, self.num_inputs - 1)
        self.matrix[cn,
        self.num_inputs:self.num_inputs + self.num_outputs] = 1
        if enforce_bias_link:
            self.matrix[self.num_inputs - 1,
            self.num_inputs:self.num_inputs + self.num_outputs] = 1  # bias
    else:
        #         self.matrix[0:self.num_inputs,
        #                     self.num_inputs:self.num_inputs+self.num_outputs] = 1

        # connect every input to every hidden
        nh = self.max_nodes - (self.num_inputs + self.num_outputs)
        for h in range(self.num_inputs + self.num_outputs, self.max_nodes):
            for i in range(self.num_inputs):
                # add_link(self.matrix, i, h,
                #         self.num_inputs, self.num_outputs, self.max_links,
                #         wm=self.weight_matrix)
                self.matrix[i, h] = 1
        # connect all hidden to outputs
        for h in range(self.num_inputs + self.num_outputs, self.max_nodes):
            for o in range(self.num_outputs):
                # add_link(self.matrix, h, self.num_inputs+o,
                #         self.num_inputs, self.num_outputs, self.max_links,
                #         wm=self.weight_matrix)
                self.matrix[h, o] = 1

    #        self.matrix[0:self.num_inputs,
    #                    self.num_inputs+self.num_outputs:self.max_nodes-self.num_inputs+self.num_outputs-1] = 1

    for i in range(self.max_nodes):
        for j in range(self.max_nodes):
            if self.matrix[i, j] == 1:
                if random_init_structural:
                    self.weight_matrix[i, j] = rnd.uniform(-max_init_weight, max_init_weight)
                else:
                    self.weight_matrix[i, j] = init_structural_with
    for i in range(self.num_inputs, num_nodes(self.matrix)):
        self.node_matrix[i - self.num_inputs, rnd.choice(arange(len(funcs)))] = 1

    f, fr, facc, faccr, nnout = test_classification(self.matrix, self.weight_matrix, self.node_matrix,
                                                    self.num_inputs, self.num_outputs, self.max_nodes)

    self.caret_row = rnd.randint(0, self.num_inputs - 1)
    self.caret_col = self.num_inputs
    self.node_caret = 0

    # these increase and when threshold is hit, then it can add new links/nodes
    self.newnode_charge = 0
    self.newlink_charge = 0

    self.prev_score = [(f, fr, facc, faccr, nnout)]
    self.best = f, fr

    self.prev_reward = [0]

    self.saved_matrix = np.copy(self.matrix)
    self.saved_weight_matrix = np.copy(self.weight_matrix)
    self.saved_caret = (self.caret_row, self.caret_col)

    observation = make_input(self.matrix, self.weight_matrix, self.node_matrix,
                             self.caret_row, self.caret_col, self.node_caret,
                             self.max_nodes, self.max_links,
                             self.num_inputs, self.num_outputs, nnout,
                             la=1, ls=0, nnc=1.0, nlc=1.0)
    return observation


GraphMakerEnv.reset = reset

# multiprocess environment or not
if 0:
    n_cpu = 8
    env = SubprocVecEnv([lambda: GraphMakerEnv(max_nodes, max_links, num_inputs, num_outputs) for i in range(n_cpu)])
else:
    env = GraphMakerEnv(max_nodes, max_links, num_inputs, num_outputs)
    env = DummyVecEnv([lambda: env])

t = 0

# Custom MLP policy of two layers of size 32 each with tanh activation function
policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[  # 'lstm',
    128,  # 256,
    128,
    128
])

tblog = 'C:/Users/spook/Dropbox/Desktop/tblog'

[shutil.rmtree(tblog + '/' + x) for x in os.listdir(tblog) if x and ('_' in x)]
time.sleep(10)

# model = PPO2(MlpPolicy, env, n_steps=240,
#              verbose=0, nminibatches=1,
#              policy_kwargs=policy_kwargs,
#              gamma=0.995,#0.99,
#              ent_coef=0.01,#0.01,
#              learning_rate=0.0002,
#              vf_coef=0.5,
#              max_grad_norm=0.5,
#              lam=0.95,
#              tensorboard_log=tblog)

# model = ACKTR(MlpPolicy, env, verbose=0, #n_steps=5,
# #              policy_kwargs=policy_kwargs,
#               tensorboard_log=tblog)

model = A2C(MlpPolicy, env, verbose=0, n_steps=64,
            #policy_kwargs=policy_kwargs,
            tensorboard_log=tblog)

# model = TRPO(MlpPolicy, env, verbose=0,# n_steps=50,
#               #policy_kwargs=policy_kwargs,
#               tensorboard_log=tblog)

# model = ACER(MlpPolicy, env, verbose=0,# n_steps=50,
#               #policy_kwargs=policy_kwargs,
#               tensorboard_log=tblog)


#################################
# Models for Box environments
#################################
#
# model = SAC(SACMlpPolicy, env, verbose=0,# n_env=1,
#                                          # n_steps=100,
#                                          # n_batch=32
#             gamma=0.995,
#             tensorboard_log=tblog
#

def learn_model():
    try:
        model.learn(total_timesteps=5000000,
                    log_interval=1000)
    except KeyboardInterrupt:
        pass

learn_model()

model.save('RL_model_1.data')


m,wm = env.get_attr('matrix')[0], env.get_attr('weight_matrix')[0]
#see_graph(m);

