import matplotlib.pyplot as plt

startx = int(600*(3/8))
starty = 50

def position2imagep(data, startgoal):
    datax = (data[0])*(3/2) + startx
    datay = (data[1])*(3/2) + starty
    return datax, datay

def plotCondition(occ, startgoal):
    fig = plt.figure()
    occ=occ.squeeze(0)
    occ=occ.squeeze(0)
    myobj = plt.imshow(occ[0, :, :])
    goalx, goaly = position2imagep(startgoal[4:], startgoal)
    plt.scatter(startx, starty)
    plt.scatter(goalx, goaly)
    for frame in occ:
        myobj.set_data(frame)
        plt.draw()
        plt.pause(0.1)

def plotData(occ, startgoal,data):
    fig = plt.figure()
    occ=occ.squeeze(0)
    occ=occ.squeeze(0)
    myobj = plt.imshow(occ[0, :, :])
    goalx, goaly = position2imagep(startgoal[4:], startgoal)
    plt.scatter(startx, starty)
    plt.scatter(goalx, goaly)
    if len(data.shape) > 1:
        for row in data:
            datax, datay = position2imagep(row, startgoal)
            plt.scatter(datax, datay, marker='*', c='#d62728')
    else:
        datax, datay = position2imagep(data, startgoal)
        plt.scatter(datax, datay, marker='*', c='#d62728')
    for frame in occ:
        myobj.set_data(frame)
        plt.draw()
        plt.pause(0.1)

def plotOrientSpeed(startgoal, data):
    fig = plt.figure()
    goalx, goaly = position2imagep(startgoal[4:], startgoal)
    plt.scatter(data[:, 2], data[:,3])
    plt.scatter(startgoal[2], startgoal[3])
    plt.scatter(startgoal[6], startgoal[7])
    
def plotAlpha(alpha):
    fig = plt.figure()
    alpha = alpha[0,:]
    alpha=alpha.reshape(6,10,60)
    
    myobj = plt.imshow(alpha[0, :, :])

    for frame in alpha:
        myobj.set_data(frame)
        plt.draw()
        plt.pause(0.3)
