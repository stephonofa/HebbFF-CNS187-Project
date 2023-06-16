import torch
import networks as nets
from gen_data import generate_image_recog_data, generate_image_recog_data_batch
#choose parameters
netType = 'HebbNet' # HebbFF or LSTM
d = 25             # input dim
N = 25             # hidden dim
force = "Anti"      # ensure either Hebbian or anti-Hebbian plasticity
trainMode = 'inf'   # train on single dataset or infinite data
R = 7               # delay interval
T = 5000            # length of dataset
save = True
img_type = "STATE"

#initialize net
if netType == 'nnLSTM':
    net = nets.nnLSTM([d,N,1])
elif netType == 'HebbNet':
    net = nets.HebbNet([d,N,1])
    if force == 'Hebb':
        net.forceHebb = torch.tensor(True)
        net.init_hebb(eta=net.eta.item(), lam=net.lam.item()) #need to re-init for this to work
    elif force == 'Anti':
        net.forceAnti = torch.tensor(True)
        net.init_hebb(eta=net.eta.item(), lam=net.lam.item())
    elif force is not None:
        raise ValueError
else:
    raise ValueError

#train
if trainMode == 'dat':
    trainData = generate_image_recog_data_batch(img_type=img_type,T=T, d=d, R=R, P=0.5, multiRep=False)
    validBatch = generate_image_recog_data(img_type=img_type,T=T, d=d, R=R, P=0.5, multiRep=False).tensors
    net.fit('dataset', epochs=4000, trainData=trainData,
            validBatch=validBatch, earlyStop=False)
elif trainMode == 'inf':
    gen_data = lambda: generate_image_recog_data_batch(img_type=img_type, T=T, d=d, R=R, P=0.5, multiRep=False)
    net.fit('infinite', gen_data, iters=4000)
else:
    raise ValueError

#optional save
if save:
    fname = '{}[{},{},1]_{}train={}{}_{}_{}.pkl'.format(
                netType, d, N, 'force{}_'.format(force) if force else '',
                trainMode, R, 'T={}'.format(T) if trainMode != 'cur' else '',
                img_type)
    net.save(fname)