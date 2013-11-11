from numpy import array, zeros
import numpy as np
import re

def read_polari(filename, nm):
    """Imports things needed from the DALTON.OUT that I need"""
    
    f = open(filename, 'r')
    second_deriv = zeros((nm,3,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('ty second derivatives',cur_line):
            line_split = cur_line.split()
            property_type = line_split[0] + line_split[1]
            finished = 1
            

    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
            
        second_deriv[mode][0][0]= mline[1]
        second_deriv[mode][0][1]= mline[2]  
        second_deriv[mode][0][2]= mline[3]
              
        second_deriv[mode][1][0]= mline[4]  
        second_deriv[mode][1][1]= mline[5]  
        second_deriv[mode][1][2]= mline[6]     
        
        
    f.close()
    return second_deriv, property_type

def read_mol_quad(filename, nm):
    """Imports things needed from the DALTON.OUT that I need"""
    
    f = open(filename, 'r')
    second_deriv = zeros((nm,3,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('quadrupole moment second derivatives',cur_line):
            line_split = cur_line.split()
            property_type = line_split[0] + line_split[1]
            finished = 1
            

    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
            
        second_deriv[mode][0][0]= mline[1]
        second_deriv[mode][0][1]= mline[2]  
        second_deriv[mode][0][2]= mline[3]
              
        second_deriv[mode][1][0]= mline[4]  
        second_deriv[mode][1][1]= mline[5]  
        second_deriv[mode][1][2]= mline[6]     
        
        
    f.close()
    return second_deriv, property_type

def read_3d_input(filename, nm):
    """Imports things needed from the DALTON.OUT that I need"""
    
    f = open(filename, 'r')
    second_deriv = zeros((nm,3,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('second derivatives for',cur_line):
            line_split = cur_line.split()
            property_type = line_split[0] + line_split[1]
            finished = 1
            
    for mode in range(nm):

        dummy = f.readline()
        dummy = f.readline()
            
        if mode != 0:
            dummy = f.readline()
            dummy = f.readline()
            dummy = f.readline()
            dummy = f.readline()
            
            dummy = f.readline()
            dummy = f.readline()
            dummy = f.readline()
            dummy = f.readline()
                                
        for i in range(3):
            dummy = f.readline()
            for j in range(3):
                mline = f.readline()
                mline = mline.split()
                second_deriv[mode][i][j]= mline[2]
    f.close()
    return second_deriv, property_type

def read_4d_input(filename, natom, nm):
    """Imports things needed from the DALTON.OUT that I need"""
    
    f = open(filename, 'r')
    second_deriv = zeros((natom,nm,3,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('second derivatives for',cur_line):
            line_split = cur_line.split()
            property_type = line_split[0] + line_split[1]
            finished = 1
            
    for atom in range(natom):

        dummy = f.readline()
        dummy = f.readline()
        dummy = f.readline()
            
        if atom != 0:
            dummy = f.readline()
            dummy = f.readline()
                                
        for mode in range(nm):
            mline = f.readline()
            mline = mline.split()
            second_deriv[atom][mode][0][0]= mline[1]
            second_deriv[atom][mode][0][1]= mline[2]  
            second_deriv[atom][mode][0][2]= mline[3]
              
            second_deriv[atom][mode][1][0]= mline[4]  
            second_deriv[atom][mode][1][1]= mline[5]  
            second_deriv[atom][mode][1][2]= mline[6]  

    f.close()
    return second_deriv, property_type
    
def read_quartic_force_field(filename, n_cord):
    """Imports the quartic force field from DALTON"""
    
    f = open(filename, 'r')
    quartic_force_field = zeros((n_cord, n_cord, n_cord, n_cord))
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('Numerical fourth derivative of energy in symmetry coordinates',cur_line):
            finished = 1
    
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    
    for D4 in range(n_cord):
        
        dummy = f.readline()
        dummy = f.readline()
        dummy = f.readline()
        for D3 in range(n_cord):
            dummy = f.readline()
            dummy = f.readline()
            for D2 in range(n_cord):
                mline = f.readline()
                values = zeros((n_cord))
                mline = mline.replace(' ', '')
                
                index1 = 0 #Before the decimal place
                index2 = 0 #Which value we are at
                
                for i in range(len(mline)):                
                    if (mline[i] == '.'):
                        values[index2] = mline[index1:i+7]
                        index2 = index2 + 1
                        index1 = i+7
                    
                for D1 in range(6):
                    quartic_force_field[D4][D3][D2][D1] = values[D1]
        
            dummy = f.readline()
            
            for D2 in range(n_cord):
                mline = f.readline()
                values = zeros((n_cord))
                mline = mline.replace(' ', '')
                
                index1 = 0 #Before the decimal place
                index2 = 0 #Which value we are at
                
                for i in range(len(mline)):                
                    if (mline[i] == '.'):
                        values[index2] = mline[index1:i+7]
                        index2 = index2 + 1
                        index1 = i+7

                for D1 in range(6):
                    quartic_force_field[D4][D3][D2][D1 + 6] = values[D1]
    f.close()
    
    return quartic_force_field
                

            


    
    
