# -*- coding: utf-8 -*-
"""
Created on Thu Mar 07 12:08:09 2013

@author: Mateon
"""
import numpy as np 
import hashlib as hl
import csv 
import  SimpleDataF as sdf;
import cPickle 

def nan_to_median( vec ) : 
    nans_idx = np.isnan( vec );
    not_nans = vec[ nans_idx == False ]
    median = np.median( not_nans )
    ret = vec.copy(); 
    ret[ nans_idx ] = median
    return ret
    
def normalize( v ) : 
    return  ( ( v - np.min( v ) )/ ( np.max(v) - np.min( v )) - 0.5 ) * 2.0

def standardize( vec ) : 
    return ( vec - vec.mean() ) / vec.std();

def sqrt_l2_normalized( vec ) : 
    """ result s, satisfies   sum( s^2 ) == 1 """
    return vec  / np.sqrt( np.sum( vec * vec ) );   
    
def integerize16( vec ) :
    """ For positive! vectors scale to [0 to 2**16 - 1 ] and round to int"""
    return ( vec/ vec.max() * ( 2**16 -1 ) ).astype( np.uint16 )
    
def log_integerize_16( vec ) : 
    """ Works for positive vectors only """
    tmp =  np.ceil( np.log10( vec / vec.max() ) * (-1000) )

def sep_train_and_validation( all_cases, target, ratio, seed, order = 'F' ) :
    np.random.seed( seed ) 
    train_idx = np.random.uniform( 0, 1 , all_cases.nrows )  < ratio
    train = all_cases.row_subset( train_idx );
    train_tgt = target[ train_idx ]
    valid = all_cases.row_subset( train_idx == False )
    valid_target = target[ train_idx == False ]
    return { "train" : train, "train_tgt" : train_tgt, 
             "train_idx" : np.nonzero( train_idx == True )[0],
             "valid" : valid, "valid_tgt" : valid_target , 
             "valid_idx" : np.nonzero( train_idx == False )[0]  } 
        
def separate_train_and_validation( train, ratio, seed   ) :
    np.random.seed( seed ) 
    train1_idx = np.random.uniform( 0, 1 , train.nrows )  < ratio
    train1 = train.row_subset( train1_idx );
    valid = train.row_subset( train1_idx == False )        
    return train1, valid  
      
def save_predictions( file_prefix, accu, predictions ) : 
    
    hash_obj = hl.sha224( str( predictions ) )
    hash_code = hash_obj.hexdigest()[0:8];

    f_out = open( "%s_%d_%s.csv" % 
                ( file_prefix, int( accu * 10000) , hash_code ), "wb" )
    output = csv.writer( f_out ) ;
    for i in xrange( len( predictions ) ) :
        output.writerow( [ int( predictions[i] ) ] ) 

    f_out.close() 
    
def precall( predicted, truth  ) : 
    ppos = sum( predicted == True )    
    tp = float( sum( (predicted == True) & ( truth == True ) ) );
    precision = tp / ppos; 
    apos = sum( truth == True )        
    recall = tp / apos;    
    return precision, recall 
    
def accuracy( predicted, truth ) : 
    return float( sum( predicted & truth ) ) / len( predicted );
    
def accu_table( predicted, truth, show = False ) : 
    assert predicted.dtype == np.bool
    assert truth.dtype == np.bool 
    
    ret = sdf.SimpleDataF() 
    ret['Pred\Actual'] = np.array( [ 'pos', 'neg', 'total' ] );
    tp = float( np.sum( predicted & truth ));
    tn = float( np.sum( (predicted == False) & ( truth == False ) ) )
    fn = float( np.sum( (predicted == False) & ( truth == True ) ) ) 
    fp = float( np.sum( predicted & ( truth == False ) ) ) 
        
    ret['pos'] = np.array( [ 100 * tp / ( tp + fn + 0.001 ) , 
                100.0 * fn / (tp + fn + 0.001),
                 int( tp + fn ) ]) 
    ret['neg'] = np.array( [ 100 * fp / ( fp + tn + 0.001 ), 
                    100 * tn / (fp + tn + 0.001),
                 int( fp + tn ) ]) 
    return ret;
    
    
import cPickle

def serialize( obj, fname ) : 
    f = open( fname, "wb" );
    cPickle.dump( obj, f );
    f.close()
    
def deserialize( fname ) : 
    f = open( fname, "rb" );
    ret = cPickle.load( f )
    f.close(); 
    return ret 
    
def sample_no_replacement( n, k ) : 
    assert k <= n 
    a = np.random.uniform( size = n );
    b = a.argsort( );
    b[0:k].sort( ) 
    return b[0:k].copy() 
    
def feats_transform( feats, type_str = None, param = 1.0 ) :
    cols = feats.cols()
    ret = sdf.SimpleDataF()
    if( type_str is None ) :
        ret =  feats 
    elif( type_str == "rank_gamma" ) : 
        for c in cols : 
            tmp = feats[c].argsort().astype( np.float64 ) / feats.nrows
            ret[c] = np.power( tmp, param )         
    elif( type_str == "rank_cut" ) :
        for c in cols : 
            tmp = feats[c].argsort().astype( np.float64 ) / feats.nrows
            ret[c] = ( tmp > param ).astype( np.float64 )         
    elif( type_str == "gamma" ) : 
        for c in cols : 
            v = feats[c]
            minim =  v.min()
            maxim = v.max() 
            tmp = ( v - minim ) / ( maxim - minim )
            ret[c] = np.power( tmp, param )
    elif( type_str == "cut" ) : 
        for c in cols : 
            v = feats[c]
            minim =  v.min()
            maxim = v.max() 
            tmp = ( v - minim ) / ( maxim - minim )
            ret[c] = ( tmp > param ).astype( np.float64 ) 
    else : 
        raise ValueError( "Do not know how to do: '%s' " % type_str )
    
    return ret
       
       
def substitute_outliers( v, min_p = 0.0, max_p = 100.0, verbose = 0 ) :     
    n = len( v )    
    v_sorted = np.sort( v )
    min_idx = int(min_p / 100.0 * n)
    min_val = v_sorted[ min_idx ]
    
    max_idx = int(max_p / 100.0 * n) - 1
    max_val = v_sorted[ max_idx ]
    
    too_small = v < min_val 
    too_large = v > max_val 

    
    ret = v.copy() 
    ret[too_small] = min_val 
    ret[too_large] = max_val 
    
    if( verbose > 0 ) : 
        print( "%d substituted by %f, %d substituted by %f " % 
            ( sum(too_small), min_val, sum(too_large), max_val ) )
    
    return ret 
    
def percentile_report( v ) : 
    n = len( v )
    v_s = np.sort( v );

    ps = [ 0, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75,
             0.90, 0.95, 0.99, 0.995, 0.998, 0.999, 1 ] 
    
    for p in ps :
        v  = v_s[ min( int( p * n ), n-1 ) ]
        print( "%.1f%% : %f" % (  p * 100, v ) )
        
        
def means_stds_per_bin( x, xbins, y ) : 
    
    assert len( x ) == len( y ) 
    
    n = len( xbins ) - 1
    means = np.zeros( n )
    means[:] = np.NaN
    stds  = np.zeros( n )
    stds[:] = np.NaN
    
    for i in xrange( n ) : 
        eps = 0.0 if i + 1 < n else 1e-6
        y_subset = y[ (x >= xbins[i]) & (x < xbins[i+1] + eps) ] 
                
        if( len( y_subset ) > 0 ) : 
            means[i] = np.mean( y_subset )
            stds[i] = np.std( y_subset )
            
    return means, stds 


def save_obj( obj, file_name ):
    fil = open( file_name, "wb" )
    cPickle.dump( obj, fil )
    fil.close() 
    
def load_obj(  file_name ) : 
    fil = open( file_name, "rb" )
    obj = cPickle.load( fil )
    fil.close()
    return obj
    
    
    
    
    
                 
    