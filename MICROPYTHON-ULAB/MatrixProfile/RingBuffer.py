#
# See https://thingsdaq.org/2023/04/18/circular-buffer-in-python/
#
import numpy as np

class RingBuffer:
    """ Class that implements a not-yet-full buffer. """
    def __init__(self, bufsize):
        self.bufsize = bufsize
        self.data = []

    class __Full:
        """ Class that implements a full buffer. """
        def add(self, x):
            """ Add an element overwriting the oldest one. """
            self.data[self.currpos] = x
            self.currpos = (self.currpos+1) % self.bufsize
        def get(self):
            """ Return list of elements in correct order. """
            return self.data[self.currpos:]+self.data[:self.currpos]

    def add(self,x):
        """ Add an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.bufsize:
            # Initializing current position attribute
            self.currpos = 0
            # Permanently change self's class from not-yet-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data

import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time

def ACAMP_1(data, my_size, m, ring):

    global Dmin
    global minind
    exc_zone = round(m / 2)
    Nb = data.shape[0]  # Number of rows in data
    s = my_size - m

    foo = 1
    for j in range(0, Nb, my_size):
        print('Chunck',foo,'of size',my_size)
        foo = foo + 1
        for jj in range(my_size):
            ring.add(data[jj+j])
            
        Dmin = np.full(s + 1, np.inf)  # Initialize Dmin with infinity
        minind = np.ones(s + 1, dtype=int)  # Initialize minind with ones
        #minind = np.zeros(s + 1, dtype=int)  # Initialize minind with zeros

        matchFlag = False

        for k in range(s):
            query = np.array(ring.get()[:m])
            target = np.array(ring.get()[k:k + m])
            
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

                sumQuery = sumQuery - np.array(ring.get()[i - 1]) + np.array(ring.get()[i + m - 1])
                sumTarget = sumTarget - np.array(ring.get()[kplusi - 1]) + np.array(ring.get()[kplusi + m - 1])

                sumSquareQuery = sumSquareQuery - np.array(ring.get()[i - 1]**2) + np.array(ring.get()[i + m - 1]**2)
                sumSquareTarget = sumSquareTarget - np.array(ring.get()[kplusi - 1]**2) + np.array(ring.get()[kplusi + m - 1]**2)

                product_me = product_me - (np.array(ring.get()[i - 1]) * np.array(ring.get()[kplusi - 1])) + (np.array(ring.get()[i + m - 1]) * np.array(ring.get()[kplusi + m - 1]))

                D = 2 * m * (1 - ((product_me - ((sumQuery * sumTarget) / m)) / 
                                    np.sqrt((sumSquareQuery - ((sumQuery**2) / m)) * 
                                            (sumSquareTarget - ((sumTarget**2) / m)))))
                if Dmin[i] > D and matchFlag:
                    minind[i] = kplusi
                    Dmin[i] = D

                if Dmin[kplusi] > D and matchFlag:
                    minind[kplusi] = i
                    Dmin[kplusi] = D

    mindist = np.sqrt(Dmin)

    return mindist, minind


# Sample usage to recreate example figure values

if __name__ == '__main__':

    # Creating ring buffer
    #x = RingBuffer(360)
    # Adding first 4 elements
    #x.add(5); x.add(10); x.add(4); x.add(7)
    # Displaying class info and buffer data
    #print(x.__class__, x.get())

    # Creating fictitious sampling data list
    #data = [1, 11, 6, 8, 9, 3, 12, 2]

    # Adding elements until buffer is full
    #for value in data[:6]:
    #    x.add(value)
    # Displaying class info and buffer data
    #print(x.__class__, x.get())

    # Adding data simulating a data acquisition scenario
    #print('')
    #print('Mean value = {:0.1f}   |  '.format(np.mean(x.get())), x.get())
    #for value in data[6:]:
    #    x.add(value)
    #    print('Mean value = {:0.1f}   |  '.format(np.mean(x.get())), x.get())

    # Size of the ring buffer
    my_size = 360
    # Motif length
    mm = 74
    # Number of windows of size my_size.
    chunk_size = 14

    from numpy import genfromtxt    
    data = genfromtxt('TourPerret.csv', delimiter=';',comments='#')
    humidity = data[:,3:4]# get humidity
    # skip the first element which corresponds to the header
    # and flatten the input for compatibility with ACAMP_1
    humidity=humidity[1:].reshape(-1)

    ringbuffer = RingBuffer(my_size)

    #
    # Normalize data to fit in [0:1]
    #
    x_norm = (humidity-np.min(humidity))/(np.max(humidity)-np.min(humidity))

    st = time.time()
    [mindist_acamp, minind_acamp] = ACAMP_1(x_norm[0:chunk_size*my_size], my_size, mm, ringbuffer)
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time ACAMP_1:', elapsed_time, 'seconds')
    # Just in case we need it
    #my_input = np.array([x for x in map(lambda el: [el], x_norm)])
    mindist_acamp = 0.60*mindist_acamp
    
    #
    # Plot
    #
    #plt.plot(x_norm[0:my_size],'red',label="Input")
    plt.plot(x_norm[(chunk_size - 1)*my_size:chunk_size*my_size],'red',label="Input")
    plt.plot(mindist_acamp,'cyan',label="ACAMP")
    plt.legend(loc="upper left")
    plt.title("Matrix Profile")
    plt.show()
    print('----------')
        
