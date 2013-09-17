# -*- coding: utf-8 -*-
"""
Created on Thu Jul 04 18:38:42 2013

@author: Mateon
"""

class CubicInterpolator( object ) : 
    def __init__( self, z, a ) :         
        """  Construct a piece-wise cubic f(x) given n-nodes z and 
        and n-by-4 
        having the form: 
         f(x) = a_3 (x - z[i])**3 + a_0 (x - z[i+1])**3 
            + a_2 ( x- z[i])**2*(x-z[i+1]) + a_1(x-z[i]) ** 2 * ()
        """
        self.z = z
        self.a = a 
    #def evaluate( self, x ) : 
        
    def n_nodes( self ) : 
        return len( self.z )
        

def smooth_fit_cubic_1( z, ny, nd ) : 
    """ Construct a piece-wise cubic f(x), given values and derivatives
        at n nodes. Satisfying f'(z[i]-) = f'(z[i]+) for each node 
        Inputs are:
        z : nodes  z(i) is the x coordinate of the i-th node
        ny : desired values for f at the nodes; ny[i] = f(z(i))
        nd : desired values for f' at the nodes; nd[i] = f'(z(i))    
        
        For x in [z[i], z[i+1]]  f(x) is given by 
        
       
    
    """
    