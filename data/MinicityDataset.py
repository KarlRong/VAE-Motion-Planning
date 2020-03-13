import sqlite3
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def position_reformed(data, startgoal):
    datax = (data[0]-startgoal[0])*(3/2) + startx
    datay = (data[1]-startgoal[1])*(3/2) + starty
    return datax, datay

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

class TrafficDataset(Dataset):
    def __init__(self, dbpath, train=True, ratio_test=0.8, num_sce=100, num_data=25):
        self.con = sqlite3.connect(dbpath, detect_types=sqlite3.PARSE_DECLTYPES)
        self.path = dbpath
        self.cur = self.con.cursor()
        self.cur.execute("select id from minicity")
        self.idlist = self.cur.fetchall()
        self.cur.execute("select data from minicity where id = " +  str(1))
        self.dataperow = int(len(self.cur.fetchone()[0]) / 2)
        numTrain = int(ratio_test * len(self.idlist))
        if (train):
            self.filelist = self.idlist[:numTrain]
        else:
            self.filelist = self.idlist[numTrain:]
            
    def __len__(self):
        return len(self.idlist) * self.dataperow
    
    def __getitem__(self, idx):
        rowid = idx // self.dataperow
        dataid = idx % self.dataperow
        self.cur.execute("select id, startgoal, occ, data, egoid from minicity where id = " +  str(rowid+1))
        results = self.cur.fetchone()
        start_goal = results[1].astype(np.single)
        observed = results[2].astype(np.single)
        traj = results[3][0:self.dataperow].astype(np.single)
        time_prob = results[3][self.dataperow].astype(np.single)[5]
        data = results[3][self.dataperow + dataid].astype(np.single)
        egoid = results[4]
        
        start_goal[4] = start_goal[4] - start_goal[0]
        start_goal[5] = start_goal[5] - start_goal[1]
        
        traj[:, 0] = traj[:, 0] - start_goal[0]
        traj[:, 1] = traj[:, 1] - start_goal[1]
        data[0] = data[0] - start_goal[0]
        data[1] = data[1] - start_goal[1]
        traj = traj[:,0:4]/50
        data = data[0:4]/50
        
        for i in range(0, int(len(observed)/2)):
            observed[i, :] = observed[2*i, :, ]
        
        observed = observed[0:int(len(observed)/2), :]
        start_goal[0] = 0
        start_goal[1] = 0
        start_goal = start_goal/50
        
        sample = {'start_goal': start_goal,
                              'traj': traj,
                             'observation': observed,
                             'data': data,
                             'egoid': egoid,
                             'timeprob':time_prob}
        return sample