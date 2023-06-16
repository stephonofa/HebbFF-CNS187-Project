import matplotlib.pyplot as plt
import numpy as np
import joblib

from net_utils import load_from_file
import plotting
from gen_data import generate_mnist_recog_data

import seaborn as sns
sns.set(font='Arial',
        font_scale=7/12., #default size is 12pt, scale down to 7pt
        palette='Set1',
        rc={'axes.axisbelow': True,
            'axes.edgecolor': 'lightgrey',
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': 'dimgrey',
            'axes.spines.right': False,
            'axes.spines.top': False,
            'text.color': 'dimgrey', #e.g. legend

            'lines.solid_capstyle': 'round',
            'legend.facecolor': 'white',
            'legend.framealpha':0.8,

            'xtick.bottom': True,
            'xtick.color': 'dimgrey',
            'xtick.direction': 'out',

            'ytick.color': 'dimgrey',
            'ytick.direction': 'out',
            'ytick.left': True,

             'xtick.major.size': 2,
             'xtick.major.width': .5,
             'xtick.minor.size': 1,
             'xtick.minor.width': .5,

             'ytick.major.size': 2,
             'ytick.major.width': .5,
             'ytick.minor.size': 1,
             'ytick.minor.width': .5
            }
        )

def mm2inch(*args):
    return [x/25.4 for x in args]

def format_and_save(fig, fname, w=None, h=None):
    if w is not None or h is not None:
        fig.set_size_inches(*mm2inch(w, h))
    fig.tight_layout()
    fig.savefig(fname)

#%% Fig 3: anti-Hebbian and continual
#(a) Train on single dataset

N = d = 25
ax = plotting.plot_loss_acc('HebbNet[{},{},1]_forceAnti_train=dat3_T=5000_mnist.pkl'.format(d,N), chance=2./3)
format_and_save(ax[0].get_figure(), f'mnist_train_val/AntiHebb_d={d}_N={N}_R=3_iters=4000.pdf', w=83, h=100)

#(b) Hebb inf, (c) Anti inf

N = d = 25
for force in ['Anti']:
    ax = None
    for R in [3,6]:
        fname = 'HebbNet[{},{},1]_force{}_train=inf{}_T={}_mnist.pkl'.format(d,N,force,R, 5000)
        net = load_from_file(fname)
        label = f'$R_{{train}}={R}$ \n $\lambda={net.lam.data.item():.2f}$, $\eta={net.eta:.2f}$'
        gen = plotting.get_generalization(fname, d=d, T=5000, upToR=50, stopAtR=50)
        ax = plotting.plot_generalization(*gen, chance=2./3,  xscale='linear', label=label, ax=ax)
    format_and_save(ax[0].get_figure(), './mnist_plots/accuracy_[{},{},1]_force={}.pdf'.format(d, N, force), w=58, h=70)

#%% Fig 4: Mechanism
# Hidden activity, weight matrix, W1x+b1 histogram for various R_train
for d in range(25,151,25):
    N = d
    ax = [plt.subplots(3,1, sharex='col')[1] for _ in range(3)]
    data1 = generate_mnist_recog_data(T=5000, R=1, d=d, P=0.5, multiRep=False)
    for i,Rtrain in enumerate([1,7,14]):
        fname = 'HebbNet[{},{},1]_force{}_train=inf{}_T={}_mnist.pkl'.format(d,N,force,Rtrain, 5000)
        net = load_from_file(fname)

        res = plotting.get_evaluation_result(fname, data1, R=1)
        h = res['h'][:20]
        out = res['out'][:20]
        isFam = res['data'].tensors[1].bool()[:20]
        plotting.plot_hidden_activity(h, isFam, out, ax=ax[0][i])
        ax[0][i].set_ylabel('$R_{{train}}={}$\n\n Neuron'.format(Rtrain))

        plotting.plot_weight(net.w1.detach(), ax=ax[1][i])

        data = generate_mnist_recog_data(T=5000, R=Rtrain, d=d, P=0.5, multiRep=False)
        Wxb = plotting.get_evaluation_result(fname, data, R=Rtrain)['Wxb']
        ax[2][i].hist(Wxb.flatten().numpy(), bins=50, density=True, histtype='step', align='mid', color='black')
        ax[2][i].set_ylabel('Probability')
    ax[2][i].set_xlabel('$W_1x(t)+b_1$')
    [a.set_xlabel('') for a in ax[0][0:2]]
    ax[1][-1].set_xlabel(' ')

    format_and_save(ax[0][0].get_figure(), f'mechanism_187_mnist/mechanism_Rtrain={[1,7,14]}_d={d}_N={N}_h.pdf', w=200, h=200)
    format_and_save(ax[1][0].get_figure(), f'mechanism_187_mnist/mechanism_Rtrain={[1,7,14]}_d={d}_N={N}_W1.pdf', w=45, h=100)
    format_and_save(ax[2][0].get_figure(), f'mechanism_187_mnist/mechanism_Rtrain={[1,7,14]}_d={d}_N={N}_Wxb.pdf', w=58, h=100)
