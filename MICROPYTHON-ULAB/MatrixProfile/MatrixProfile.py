#
#
# For AAMP, see: https://github.com/anoynymREVIEW/ICDM_AAMP_ACAMP/blob/master/Code/AAMP.m
#      and https://sites.google.com/view/aamp-and-acamp/home
#
# For stump, see the documentation at https://stumpy.readthedocs.io/en/latest/api.html#stumpy.stump
#
# For scrimp_plus_plus, read the header of file scrimp.py
#
# Compute the self similarity join of time series
#
# For documentation on possible distances, see
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
#
#

from stumpy import stump
from scrimp import scrimp_plus_plus

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time

def ACAMP_1(data, m):

    exc_zone = round(m / 2)
    Nb = data.shape[0]  # Number of rows in data
    s = Nb - m
    Dmin = np.full(s + 1, np.inf)  # Initialize Dmin with infinity
    minind = np.ones(s + 1, dtype=int)  # Initialize minind with ones
    #minind = np.zeros(s + 1, dtype=int)  # Initialize minind with zeros

    matchFlag = False

    for k in range(s):
        query = data[:m]
        target = data[k:k + m]

        sumQuery = np.sum(query)
        sumTarget = np.sum(target)

        sumSquareQuery = np.sum(query**2)
        sumSquareTarget = np.sum(target**2)

        product_me = np.sum(query * target)

        D = 2 * m * (1 - ((product_me - ((sumQuery * sumTarget) / m)) / 
                                np.sqrt((sumSquareQuery - ((sumQuery**2) / m)) * 
                                        (sumSquareTarget - ((sumTarget**2) / m)))))

        if k > exc_zone:
            matchFlag = True

        if D < Dmin[0] and matchFlag:
            Dmin[0] = D
            minind[0] = k + 1

        if D < Dmin[k + 1] and matchFlag:
            Dmin[k + 1] = D
            minind[k + 1] = 1

        for i in range(1, s - k + 1):
            kplusi = k + i

            sumQuery = sumQuery - data[i - 1] + data[i + m - 1]
            sumTarget = sumTarget - data[kplusi - 1] + data[kplusi + m - 1]

            sumSquareQuery = sumSquareQuery - data[i - 1]**2 + data[i + m - 1]**2
            sumSquareTarget = sumSquareTarget - data[kplusi - 1]**2 + data[kplusi + m - 1]**2

            product_me = product_me - (data[i - 1] * data[kplusi - 1]) + (data[i + m - 1] * data[kplusi + m - 1])

            D = 2 * m * (1 - ((product_me - ((sumQuery * sumTarget) / m)) / 
                                    np.sqrt((sumSquareQuery - ((sumQuery**2) / m)) * 
                                            (sumSquareTarget - ((sumTarget**2) / m)))))
            #if D <= 0.0:
                #print(D,i,k+i,sumQuery,sumTarget,sumSquareQuery,sumSquareTarget,product_me)
                # 190.0625 183.90625 483.85546875 452.9248046875 468.138671875
                
            if Dmin[i] > D and matchFlag:
                minind[i] = kplusi
                Dmin[i] = D

            if Dmin[kplusi] > D and matchFlag:
                minind[kplusi] = i
                Dmin[kplusi] = D

    mindist = np.sqrt(Dmin)
    # Normalize the result to be positive
    #mindist = (Dmin-np.min(Dmin))/(np.max(Dmin)-np.min(Dmin))

    return mindist, minind


def AAMP(X, m):
    exc_zone = round(m / 2)
    #Nb = len(X)
    Nb = X.shape[0]  # Number of rows in data
    s = Nb - m
    Dmin = np.full(s + 1, np.finfo(float).max)
    minind = np.ones(s + 1, dtype=int)
    
    matchFlag = False
    for k in range(s):
        D = np.sum((X[:m] - X[k:m + k]) ** 2)
        
        if k > exc_zone:
            matchFlag = True
            
        if D < Dmin[0] and matchFlag:
            Dmin[0] = D
            minind[0] = k + 1
            
        if D < Dmin[k + 1] and matchFlag:
            Dmin[k + 1] = D
            minind[k + 1] = 1
        
        for i in range(1, s - k + 1):
            kplusi = k + i
            D = D - (X[i - 1] - X[kplusi - 1]) ** 2 + (X[i + m - 1] - X[kplusi + m - 1]) ** 2
            if Dmin[i] > D and matchFlag:
                minind[i] = kplusi
                Dmin[i] = D

            if Dmin[kplusi] > D and matchFlag:
                minind[kplusi] = i
                Dmin[kplusi] = D
                
    mindist = np.sqrt(Dmin)
    
    return mindist, minind

#
# Normalized Euclidean Distance (NED)
#

def AAMP_ned(X, m):
    exc_zone = round(m / 2)
    #Nb = len(X)
    Nb = X.shape[0]  # Number of rows in data
    s = Nb - m
    mm = 4*m
    Dmin = np.full(s + 1, np.finfo(float).max)
    minind = np.ones(s + 1, dtype=int)
    
    matchFlag = False
    for k in range(s):
        # Normalized Eucliean Distance (NEQ)
        D = np.sum((X[:m]/mm - X[k:m + k]/mm) ** 2)

        if k > exc_zone:
            matchFlag = True
            
        if D < Dmin[0] and matchFlag:
            Dmin[0] = D
            minind[0] = k + 1
            
        if D < Dmin[k + 1] and matchFlag:
            Dmin[k + 1] = D
            minind[k + 1] = 1
        
        for i in range(1, s - k + 1):
            kplusi = k + i
            # Normalized Eucliean Distance (NEQ)
            D = D - (X[i - 1]/mm - X[kplusi - 1]/mm) ** 2 + (X[i + m - 1]/mm - X[kplusi + m - 1]/mm) ** 2
            if Dmin[i] > D and matchFlag:
                minind[i] = kplusi
                Dmin[i] = D

            if Dmin[kplusi] > D and matchFlag:
                minind[kplusi] = i
                Dmin[kplusi] = D
                
    mindist = np.sqrt(Dmin)

    return mindist, minind

#
#  Minkowski Distance (MD)
#

def AAMP_md(X, m, p):
    exc_zone = round(m / 2)
    #Nb = len(X)
    Nb = X.shape[0]  # Number of rows in data
    s = Nb - m
    Dmin = np.full(s + 1, np.finfo(float).max)
    minind = np.ones(s + 1, dtype=int)
    
    matchFlag = False
    for k in range(s):
        # Minkowski Distance (MD)
        D = np.sum(np.abs(X[:m] - X[k:m + k]) ** p)

        if k > exc_zone:
            matchFlag = True
            
        if D < Dmin[0] and matchFlag:
            Dmin[0] = D
            minind[0] = k + 1
            
        if D < Dmin[k + 1] and matchFlag:
            Dmin[k + 1] = D
            minind[k + 1] = 1
        
        for i in range(1, s - k + 1):
            kplusi = k + i
            # Minkowski Distance (MD)
            D = D - (X[i - 1] - X[kplusi - 1]) ** p + (X[i + m - 1] - X[kplusi + m - 1]/mm) ** p
            if Dmin[i] > D and matchFlag:
                minind[i] = kplusi
                Dmin[i] = D

            if Dmin[kplusi] > D and matchFlag:
                minind[kplusi] = i
                Dmin[kplusi] = D
                
    mindist = Dmin ** (1/p)

    return mindist, minind


if __name__ == "__main__":
    print('Analyze the outside relative humidity of Tour Perret Dataset')
    # Opening the JSON file
    df = pd.read_json('ems-tourperret.ndjson', lines=True)
    Nrows = int(df.shape[0])

    # Use Itertuples to iterate over payload keys
    data = []
    for i, row in enumerate(df.itertuples(index=False)):

        val = row[df.columns.get_loc('decoded')]

        if 'payload' in val:
            res = []
            for key in val['payload']:
                #print('\t',key,':',val['payload'][key])
                res.append(val['payload'][key])

            data.append(res)
        else:
            Nrows = Nrows - 1
            #print('No payload',val,Nrows)
        
    #
    # Ready to process the data?
    #
    data = np.array(data)
    # convert array into dataframe 
    DF = pd.DataFrame(data) 
    # save the dataframe as a csv file 
    DF.to_csv("TourPerret.csv",sep=";",header=["accMotion","digital","humidity","pulseAbs","temperature","vdd","waterleak","x","y","z"])
    my_size = 360
    #mm = 54
    mm = 75

    # We choose the third column that contains outside relative humidity
    X = data[:, 2]
    # We choose the first lines of the dataset
    Y = X [:my_size]

    #
    # Normalize data to fit in [0:1]
    #
    x_norm = (Y-np.min(Y))/(np.max(Y)-np.min(Y))

    # Go
    st = time.time()
    [mindist, minind] = AAMP(2+x_norm, mm)#int(mm/3))
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time AAMP:', elapsed_time, 'seconds')
    ysmoothed = gaussian_filter1d(mindist, sigma=2)

    st = time.time()
    [mindist_ned, minind_ned] = AAMP_ned(2+x_norm, mm)#int(mm/3))
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time AAMP_ned:', elapsed_time, 'seconds')
    aamp_ned_norm = (mindist_ned-np.min(mindist_ned))/(np.max(mindist_ned)-np.min(mindist_ned))

    st = time.time()
    [mindist_md, minind_md] = AAMP_md(x_norm, mm, 3)
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time AAMP_mp:', elapsed_time, 'seconds')
    aamp_md_norm = (mindist_md - np.min(mindist_md))/(np.max(mindist_md)-np.min(mindist_md))

    st = time.time()
    [mindist_acamp, minind_acamp] = ACAMP_1(x_norm, mm)
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time ACAMP_1:', elapsed_time, 'seconds')
    # Just in case we need it
    #my_input = np.array([x for x in map(lambda el: [el], x_norm)])
    mindist_acamp = 1.2*mindist_acamp
    
    st = time.time()
    matrix_profile = stump(x_norm, mm)
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time stump:', elapsed_time, 'seconds')
    mp_stump = 1.6 * matrix_profile[:,0] # See https://stumpy.readthedocs.io/en/latest/api.html#stumpy.stump

    st = time.time()
    step_size = 0.129
    mp, mpidx = scrimp_plus_plus(x_norm, mm, step_size)
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time scrimp++:', elapsed_time, 'seconds')

    #
    # Plot
    #
    plt.plot(x_norm,'red',label="Input")
    plt.plot(mindist,'blue',label="AAMP")
    plt.plot(aamp_ned_norm,'black',label="AAMP_ned")
    plt.plot(aamp_md_norm,'magenta',label="AAMP_md")
    plt.plot(mindist_acamp,'cyan',label="ACAMP")
    #plt.plot(ysmoothed,'y',label="Smooth AAMP")
    plt.plot(mp,'g',label="Scrimp++")
    plt.plot(mp_stump,'orange',label="Stump")
    plt.legend(loc="upper left")
    plt.title("Matrix Profile")
    plt.show()
