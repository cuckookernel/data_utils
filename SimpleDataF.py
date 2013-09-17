# -*- coding: utf-8 -*-
"""
Created on Thu Mar 07 09:31:17 2013

@author: Mateon
"""

from collections import OrderedDict as Odict

import sys
import numpy as np;

def make_new() : 
    return SimpleDataF(    )
    
class DataFiterator( object ) : 
    def __init__( self, df, row = 0 ) : 
        if( row - 1 > df.nrows ) : 
            raise StopIteration 
        self.df = df 
        self.row = row -1 
    def __setitem__( self, col, val ) : 
        self.df[ col ][ self.row ] = val
    def __getitem__( self, col ) : 
        return self.df[col][self.row ]
        
    def set( self, **kwargs  ) : 
        for col, val in kwargs.iteritems() : 
            self.df[col][self.row] = val
    def next( self ) : 
        self.row += 1 
        if( self.row >= self.df.nrows ) : 
            raise StopIteration
        else : 
            return self 

class SimpleDataF( object )  :
    def __init__( self, data = None ,  headers = None, ) :

        if( data is None ) : 
            data = Odict() 
            self.nrows = 0 
        
        if( type( data ) == dict or type(data) == Odict  ) :
            # self.cols = data.keys()            
            self.data_ = Odict() 
            self.first_k  = None
            for col, value in data.iteritems() : 
                self[ col ] = value  
        elif( type( data ) == np.ndarray  ) : 
            assert data.shape[1] == len( headers )
            self.data_ = Odict() 
            self.first_k = None 
            for idx, hdr in zip ( xrange(data.shape[1]), headers ) : 
                self[ hdr ] = data[ : , idx ]
        else : 
            raise ValueError( "Instantiating SimpleDataF from %s is not implemented"
            % type( data )  )
            
#        if( type(data) == np.array ) : 
#            if( len( data.shape ) > 2 ) : 
#                raise RuntimeError( "Don't know what to do with array that is not a rank 2 array " )
#                self.nrows = data.shape[0]                
#                self.ncols = data.shape[1]
    def make_new( self ) : 
        """ New and empty data frame of the same type """
        return type( self )( );
                
    def array( self, cols = None, dtype = None, order = 'C' ) : 
        """ Return a numpy array with shape nrows x len( cols ) """
        if( not cols ) :
            cols = self.cols()

        if( not dtype ) :
            dtype0 = self.data_[cols[0]].dtype     
        else :
            dtype0 = dtype 
            
        ret = np.zeros( ( self.nrows, len( cols ) ),
                       dtype = dtype0, order = order  )
        
        for i in xrange( 0, len( cols )) : 
            ret[ : , i ] = self.data_[cols[i]]
            
        return ret 
        
    def keys( self ) : 
        return self.data_.keys() 
        
    def cols( self ) : 
        return self.data_.keys()
        
    def ncols( self ) : 
        return len( self.data_.keys() ) 
          
    
    def __getitem__( self, col ) :
        """ return reference to column """
        return self.data_[ col ]
            
    def __setitem__( self, col, value ) : 
        """ set a column to an array or list (will be converted 
        to np.ndarray """ 
        # print( "type of np: %s  " % (  type( np ) ) )
        if( type( value ) != np.ndarray and type( value ) != list ) :
            raise ValueError( "Col %s of type %s, expected numpy.ndarray or list" 
            % (col, type(col)) )
                
        value1 = np.array( value )                    
            
        if( not self.first_k ) :
            self.first_k = col
            self.nrows = np.size( value1 )
        else :
            if( np.size( value1 ) != self.nrows ) : 
                raise ValueError( "While adding col '%s', found %d elements, expected %d (= len. of col %s) " 
                % ( col, np.size(value1), self.nrows, self.first_k )  )
                
        self.data_[ col ] = value1
        self.data_[ col ].reshape( ( self.nrows, ) )
        
    def __iter__( self ) : 
        return DataFiterator( self, 0 )
            
    def select( self, cols, copy = True ) : 
        if(  cols is None ) :
            cols = self.cols() 
        
        ret = self.make_new(); 
        for c in cols : 
            if( copy ) :
                ret[c] = self.data_[c];
            else : 
                ret[c] = self.data_[c].copy();
        
        return ret
        
    def copy( self ) : 
        return self.select( None, copy = True )
        
    def select_not( self, cols = None, copy = True ) : 
        colsInclude = filter( lambda c : c not in cols,  self.cols() );
        return self.select( colsInclude, copy = copy );
        
    def drop1 ( self, col, copy = True, in_place = False ) :
        if( in_place ) : 
            if( col in self.cols() ) : 
                del self.data_[ col ]
            return self
        else : 
            colsInclude = filter( lambda c : c != col,  self.cols() );
            return self.select( colsInclude, copy = copy );
        
    def row_subset( self, row_index, copy = True, in_place = False ) :
        ret = self if( in_place ) else self.make_new();
         
        for  c in self.cols() :
            col = self[c]
            # print( "c = %s col.shape = %s" % ( c, col.shape ))
            ret[ c ] =  ( col[ row_index ].copy() if( copy ) else
                col[ row_index ] );            
        return ret 
        
    def row( self, idx ) : 
        """ Get a single row as dict """
        cols = self.cols()
        return dict(  zip( cols, map( lambda c : self[c][idx], cols ) ) ) 
        
    def show_head( self, n = 10 ) : 
        row_index = range( 0, n ); 
        to_show = self.row_subset( row_index, copy = False ); 
        print( to_show );
    
    def show_tail( self, n = 10 ) : 
        row_index = range( self.nrows - n , self.nrows ); 
        to_show = self.row_subset( row_index, copy = False ); 
        print( to_show );
            
    def union( self,  other, in_place = False ) : 
        
        if( set(  self.cols() )  != set( other.cols() ) )  : 
            raise ValueError( "this  cols (%d): %s\nother cols (%d) : %s \n" % 
                ( len( self.cols() ), repr( set(self.cols()) ), 
                 len( other.cols() ), repr( set(other.cols()) ) ) ) 
        
        ret = self if in_place else self.make_new();
                
        for c in self.cols() : 
            ret[c] = np.hstack([ self[c] , other[c] ]); 
            
        return ret;
    def zip( self, other, in_place = False, 
                collision = "error", postfix = "_2" ) : 
        ret = self if in_place else self.copy()
        for c in other.cols() : 
            new_col = c
            if( c in ret.cols() ) : 
                if( collision == "error" ) :
                    raise ValueError( "Column %s from 'other' is already in self" % c )
                elif( collision == "warn" ) : 
                    sys.stderr.write( "Column %s from 'other' is already in self. Will overwrite" );
                elif( collision == "overwrite" ) : 
                    pass; 
                elif( collision == "postfix" ) : 
                    new_col = c + postfix 
            ret[new_col] = other[c]   
        return ret                                 
        
    def apply_each_col( self, fun, in_place = False ) :
        ret = self if in_place else  self.make_new(); 
        for c in self.cols() : 
            ret[c] = fun( self[c] )
        return ret 
    def iter_cols( self ) : 
        for c in self.cols() : 
            yield self[c]        
    def summary(self) : 
        ret_data = Odict() 
        
        cols = self.cols();
        n_cols = len( cols );
        
        ret_data[ "col" ] = np.array( cols  )
        for name,fun in [ ( "min",  np.min), ("max", np.max), 
                         ("mean",  np.mean), ( "median", np.median), 
                        ( "std", np.std ) ] : 
            ret_data[name] = np.ones( n_cols ) * np.nan
            for c, i in zip(  cols, xrange( n_cols ) ) : 
                col = self[c]
                if( isfloat( col ) ) :
                    col1 = col[ np.isnan( col ) == False ]
                    ret_data[ name ][i] = fun( col1 )
                
           
        ret_data[ "n_nans" ] = np.zeros( n_cols, dtype = "int64" );
        
        for c, i in zip(  cols, xrange( n_cols ) ) : 
            if( isfloat( self[c] ) ):
                ret_data[ "n_nans" ][i] =  np.isnan( self[c]).sum()                        
                
        return SimpleDataF( ret_data )
        
    def __str__( self , sep = " ") : 
        rows = [ None ] * ( self.nrows + 1 ) 
        rows[0] = sep.join( map( lambda c : "%12s" % c , self.cols() ) )

        for i in xrange( self.nrows ) : 
            
            strs = map( lambda c : "%12s" %  fmt(self[c][i]) , self.cols() )
            rows[ i + 1 ] = sep.join( strs )  
        
        return "\n".join( rows )
    def __repr__( self ) : 
        return "<%s %08x rows %d, cols : %s >" % ( 
            type( self ).__name__, id( self ), self.nrows, self.cols() )

        
def isfloat( v ) : 
    return v.dtype == np.float64 or v.dtype == np.float32 
        
def fmt( v ) : 
    if( type(v) == float or type(v )  == np.float64  ) : 
        return "%12f" % v 
    else :
        return "%012s"% v 
   
def zeros( nrows, cols, dtypes_dict = None, dtype = np.float64,  ) : 
    ret = SimpleDataF()
    for col in cols : 
        dtype1 = dtype 
        if( dtypes_dict is not None and col in dtypes_dict ) : 
            dtype1 = dtypes_dict[col]
        ret[col] = np.zeros( nrows, dtype = dtype1 )
    return ret      
        
import csv 
from collections import deque

def load_csv( fname, types_map = {}, require_types = True ) :
    
    f = open( fname, "rb" )

    csv_read = csv.reader( f )    
    headers = csv_read.next()
    n_headers = len( headers )
    ci = dict(  zip( headers, xrange( 0, len( headers ))))
    
    if( require_types ) :
        for h in headers : 
            assert h in types_map, " %s not in types_map:\n%s" % ( 
                                    h, types_map )     
    
    row_count = 0; 
    data_raw = deque();    
    for row in csv_read : 
        row_count += 1;            
        assert len( row ) == n_headers, ( 
            "at row_count = %d, wrong number of pieces=%d != %d, row=\n%s" 
            % (  row_count, len(row), n_headers, " | ".join( row ) ) )  
        data_raw.append( row  )    
        
    f.close();

    ret = SimpleDataF(); 
    for h in headers :
        i = ci[ h ];
        dt = types_map.get( h );
        arr = np.array( map( lambda row : row[i], data_raw ) )
        if( dt is not None ) : 
            arr = arr.astype( dt )
        ret[h] = arr;
        
    return ret
    

            