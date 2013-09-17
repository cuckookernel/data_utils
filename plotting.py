# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:33:52 2013

@author: Mateon
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import data_utils as du 
reload( du )


def close_all( ) : 
    plt.close( "all" )


def plot_hist( x , nbins = 50, normed = 1 ) :     

    fig = plt.figure()
    ax = fig.add_subplot(111)

# the histogram of the data
    n, bins, patches = ax.hist(x, nbins, normed=normed, 
                                  facecolor='green', alpha=0.75)
                                  
    ax.set_ylim( 0, np.max(n)*1.05 )
              
    plt.show()                        
    return ax 


def plot_hist_and_ymeans( x, y, nbins = 50, normed = 1 ) :
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    n, bins, patches = ax.hist(x, nbins, normed=normed, 
                                  facecolor='green', alpha=0.75)
    ax.set_ylim( 0, np.max(n)*1.05 )
    ax.set_xlim( np.min( x ), np.max( x ))
    
    means, stds = du.means_stds_per_bin( x, bins, y ) 
    
    ax2 = ax.twinx() 
    print( ax2.get_xlim() )    
    ax2.errorbar( (bins[:-1] + bins[1:])/2 , means, yerr=stds ) 
    ax.set_xlim( np.min( x ), np.max( x ))
    print( ax2.get_xlim() )        
    ax2.set_ylim(np.nanmin( means - stds ),   np.nanmax( means + stds ))

    return ax 

ax = plot_hist_and_ymeans( w[:N_TRAIN], useful_rate, nbins = 4 )
