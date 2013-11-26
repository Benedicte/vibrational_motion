from numpy import array, zeros, add, subtract, diag
import numpy as np
import re

def read_mol_quad(filename, nm):
    """Imports things needed from the DALTON.OUT that I need"""
    
    f = open(filename, 'r')
    second_deriv = zeros((nm,3,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('Molecular quadrupole moment second derivatives',cur_line):
            finished = 1
            

    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
            
        second_deriv[mode][0][0]= mline[1]
        second_deriv[mode][0][1]= mline[2]  
        second_deriv[mode][1][1]= mline[3]
        second_deriv[mode][0][2]= mline[4]  
        second_deriv[mode][1][2]= mline[5]  
        second_deriv[mode][2][2]= mline[6]   
        
    #second_deriv_t = second_deriv[mode].transpose()
    #second_deriv_temp = add(second_deriv[mode], second_deriv_t) 
    #second_deriv[mode] = subtract(second_deriv_temp , diag(second_deriv[mode].diagonal()))   
    
    f.close()
    return second_deriv, "MOLQUAD"
    
def read_magnet(filename, nm):
    
    f = open(filename, 'r')
    second_deriv_magnet = zeros((nm,3,3))
    second_deriv_g = zeros((nm,3,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('Magnetizability tensor second derivatives',cur_line):
            finished = 1
    
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
     
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
        second_deriv_magnet[mode][0][0]= mline[1]
        second_deriv_magnet[mode][0][1]= mline[2]  
        second_deriv_magnet[mode][1][1]= mline[3]
              
        second_deriv_magnet[mode][0][2]= mline[4]  
        second_deriv_magnet[mode][1][2]= mline[5]  
        second_deriv_magnet[mode][2][2]= mline[6]
        
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline() 
    dummy = f.readline()
    dummy = f.readline()
     
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
        second_deriv_g[mode][0][0]= mline[1]
        second_deriv_g[mode][0][1]= mline[2]  
        second_deriv_g[mode][1][1]= mline[3]
        second_deriv_g[mode][0][2]= mline[4]  
        second_deriv_g[mode][1][2]= mline[5]  
        second_deriv_g[mode][2][2]= mline[6]
        
        second_deriv_t = second_deriv_g[mode].transpose()
        second_deriv_temp = add(second_deriv_g[mode], second_deriv_t) 
        second_deriv_g[mode] = subtract(second_deriv_temp, diag(second_deriv_g[mode].diagonal())) 
        
    f.close()
    
    return second_deriv_magnet, second_deriv_g    

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
        second_deriv[mode][1][1]= mline[3]
        second_deriv[mode][0][2]= mline[4]  
        second_deriv[mode][1][2]= mline[5]  
        second_deriv[mode][2][2]= mline[6]   
        
    f.close()
    return second_deriv, property_type

def read_molquad1(filename, nm):
    """Imports things needed from the DALTON.OUT that I need"""
    
    f = open(filename, 'r')
    second_deriv = zeros((nm,3,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('Molecular quadrupole moment second derivatives',cur_line):
            line_split = cur_line.split()
            property_type = line_split[0] + line_split[1]
            finished = 1
            

    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    print "molquad"
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
            
        second_deriv[mode][0][0]= mline[1]
        second_deriv[mode][0][1]= mline[2]  
        second_deriv[mode][1][1]= mline[3]
        second_deriv[mode][0][2]= mline[4]
        second_deriv[mode][1][2]= mline[5]  
        second_deriv[mode][2][2]= mline[6]   
        print second_deriv[mode]
        
        second_deriv_t = second_deriv[mode].transpose()
        second_deriv_temp = add(second_deriv[mode], second_deriv_t) 
        second_deriv[mode] = subtract(second_deriv_temp , diag(second_deriv[mode].diagonal()))   
        
    f.close()

    return second_deriv, property_type

def read_spinrot(filename, natom, nm):
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
            second_deriv[atom][mode][2][0]= mline[7]
            second_deriv[atom][mode][2][1]= mline[8]  
            second_deriv[atom][mode][2][2]= mline[9] 

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
            second_deriv[atom][mode][1][1]= mline[3]
            second_deriv[atom][mode][0][2]= mline[4]  
            second_deriv[atom][mode][1][2]= mline[5]  
            second_deriv[atom][mode][2][2]= mline[6] 

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
                

            


    
    
