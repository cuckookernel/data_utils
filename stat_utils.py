# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:53:04 2013

@author: Mateon
"""

import numpy as np


def wgt_std(arrin, weights_in, inputmean=None ):
    """
    NAME:
      wgt_std()
      
    PURPOSE:
      Calculate the weighted standard deviation of
      an input array.  

    CALLING SEQUENCE:
     wmean = wgt_std(arr, weights, inputmean=None)
    
    INPUTS:
      arr: A numpy array or a sequence that can be converted.
      weights: A set of weights for each elements in array.
    OPTIONAL INPUTS:
      inputmean: 
          If not None, an input mean value, 
          around which them mean is calculated. If not provided

    OUTPUTS:
      wstd: weighted standard deviation

   """
    
    # no copy made if they are already arrays
    arr = np.array(arrin, ndmin=1, copy=False)
    
    # Weights is forced to be type double. All resulting calculations
    # will also be double
    weights = np.array(weights_in, ndmin=1, dtype='float64', copy=False)
  
    wtot = weights.sum()
        
    # user has input a mean value
    if inputmean is None:
        wmean = ( weights*arr ).sum()/wtot
    else:
        wmean=float(inputmean)
    
    wvar = ( weights*(arr-wmean)**2 ).sum()/wtot
    return  np.sqrt(wvar)