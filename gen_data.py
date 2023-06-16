import numpy as np
import torch
import pickle

from torch.utils.data import TensorDataset

def generate_mnist_recog_data(T=2000, d=25, R=1, P=0.5, interleave=True, multiRep=False):
    """Generates mnist recognition dataset sequence of (x,y) tuples. 
    x[t] is a d-dimensional binary vector representing an mnist image, 
    y[t] is 1 if x[t] has appeared in the sequence x[0] ... x[t-1], and 0 otherwise
    
    if interleave==False, (e.g. R=3) ab.ab.. is not allowed, must have a..ab..b.c..c (dots are new samples)
    if multiRep==False a pattern will only be (intentionally) repeated once in the trial
    
    T: length of trial
    d: length of x
    R: repeat interval
    P: probability of repeat
    """  
    if np.isscalar(R):
        Rlist = [R]
    else:
        Rlist = R
    
    with open(f"./loaded_encodings/mnist_d={d}_encodings.pkl", "rb") as pkl_file:
        all_encodings = pickle.load(pkl_file)
        assert(type(all_encodings) == dict)
    encoding_labels = []
    
    data = []
    repeatFlag = False
    r=0 #countdown to repeat
    for t in range(T): 
        #decide if repeating
        R = Rlist[np.random.randint(0, len(Rlist))]
        if interleave:
            repeatFlag = np.random.rand()<P
        else:
            if r>0:
                repeatFlag = False
                r-=1
            else:
                repeatFlag = np.random.rand()<P 
                if repeatFlag:
                    r = R
                
        #generate datapoint
        if t>=R and repeatFlag and (multiRep or data[t-R][1].round()==0):
            label = encoding_labels[t-R]
            codes = all_encodings[str(label)]
            idx = np.random.choice(np.arange(len(codes)))
            code = codes[idx]
            encoding_labels.append(label)
            x = np.array(code)
            y = 1
            del codes[idx]
        else:
            label = np.random.choice(np.arange(10))
            codes = all_encodings[str(label)]
            idx = np.random.choice(np.arange(len(codes)))
            code = codes[idx]
            encoding_labels.append(label)
            x = np.array(code)         
            y = 0
            del codes[idx]
            
                     
        data.append((x,np.array([y]))) 
        
    return data_to_tensor(data)


def generate_image_recog_data(img_type: str, T=2000, d=25, R=1, P=0.5, interleave=True, multiRep=False):
    """Generates image recognition dataset sequence of (x,y) tuples. 
    x[t] is a d-dimensional binary vector representing an mnist image, 
    y[t] is 1 if x[t] has appeared in the sequence x[0] ... x[t-1], and 0 otherwise
    
    img_type is either EXEMPLAR, OBJECTSALL, or STATE
    if interleave==False, (e.g. R=3) ab.ab.. is not allowed, must have a..ab..b.c..c (dots are new samples)
    if multiRep==False a pattern will only be (intentionally) repeated once in the trial
    
    T: length of trial
    d: length of x
    R: repeat interval
    P: probability of repeat
    """  
    if np.isscalar(R):
        Rlist = [R]
    else:
        Rlist = R
    
    all_encodings = np.load(f"./loaded_encodings/{img_type.upper()}_d={d}_encodings.npy")
    
    data = []
    repeatFlag = False
    r=0 #countdown to repeat
    for t in range(T): 
        #decide if repeating
        R = Rlist[np.random.randint(0, len(Rlist))]
        if interleave:
            repeatFlag = np.random.rand()<P
        else:
            if r>0:
                repeatFlag = False
                r-=1
            else:
                repeatFlag = np.random.rand()<P 
                if repeatFlag:
                    r = R
                
        #generate datapoint
        if t>=R and repeatFlag and (multiRep or data[t-R][1].round()==0):
            x = data[t-R][0]
            y = 1
        else:
            x = all_encodings[np.random.choice(np.arange(len(all_encodings)))]        
            y = 0
            
                     
        data.append((x,np.array([y]))) 
        
    return data_to_tensor(data)


def generate_mnist_recog_data_batch(T=2000, batchSize=1, d=25, R=1, P=0.5, interleave=True, multiRep=False, device='cpu'):
    """Faster version of recognition data generation. Generates in batches and uses torch directly    
    Note: this is only faster when approx batchSize>4
    """  
    if np.isscalar(R):
        Rlist = [R]
    else:
        Rlist = R
    
    with open(f"./loaded_encodings/mnist_d={d}_encodings.pkl", "rb") as pkl_file:
        all_encodings = pickle.load(pkl_file)
        assert(type(all_encodings) == dict)

    x = np.zeros(shape=(T, batchSize, d))
    for i in range(T):
        for j in range(batchSize):
            label = np.random.choice(np.arange(10))
            codes = all_encodings[str(label)]
            idx = np.random.choice(np.arange(len(codes)))
            code = codes[idx]
            x[i,j] = code
            del codes[idx]
    
    x = torch.Tensor(x)
    
    y = torch.zeros(T,batchSize, dtype=torch.bool, device=device)
    
    for t in range(max(Rlist), T):
        R = Rlist[np.random.randint(0, len(Rlist))] #choose repeat interval   
        
        if interleave:
            repeatMask = torch.rand(batchSize)>P
        else:
            raise NotImplementedError
        
        if not multiRep:
            repeatMask = repeatMask*(~y[t-R]) #this changes the effective P=n/m to P'=n/(n+m)
          
        x[t,repeatMask] = x[t-R,repeatMask]            
        y[t,repeatMask] = 1
        
    y = y.unsqueeze(2).float()

    return TensorDataset(x, y)


def generate_image_recog_data_batch(img_type: str, T=2000, batchSize=1, d=25, R=1, P=0.5, interleave=True, multiRep=False, device='cpu'):
    """Faster version of recognition data generation. Generates in batches and uses torch directly    
    Note: this is only faster when approx batchSize>4
    """  
    if np.isscalar(R):
        Rlist = [R]
    else:
        Rlist = R
    
    all_encodings = np.load(f"./loaded_encodings/{img_type.upper()}_d={d}_encodings.npy")

    x = np.zeros(shape=(T, batchSize, d))

    for i in range(T):
        for j in range(batchSize):
            x[i,j] = all_encodings[np.random.choice(np.arange(len(all_encodings)))]
    
    x = torch.Tensor(x)
    
    y = torch.zeros(T,batchSize, dtype=torch.bool, device=device)
    
    for t in range(max(Rlist), T):
        R = Rlist[np.random.randint(0, len(Rlist))] #choose repeat interval   
        
        if interleave:
            repeatMask = torch.rand(batchSize)>P
        else:
            raise NotImplementedError
        
        if not multiRep:
            repeatMask = repeatMask*(~y[t-R]) #this changes the effective P=n/m to P'=n/(n+m)
          
        x[t,repeatMask] = x[t-R,repeatMask]            
        y[t,repeatMask] = 1
        
    y = y.unsqueeze(2).float()

    return TensorDataset(x, y)

#%%############
### Helpers ###   
###############
def data_to_tensor(data, y_dtype=torch.float, device='cpu'):
    '''Convert from list of (x,y) tuples to TensorDataset'''
    x,y = zip(*data)
    return TensorDataset(torch.as_tensor(x, dtype=torch.float, device=device), 
                   torch.as_tensor(y, dtype=y_dtype, device=device))