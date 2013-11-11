from numpy import array, zeros, vstack, dot, identity, sqrt, set_printoptions, compress, reshape, multiply, divide, add, subtract, diag, absolute, sort, argsort, fliplr
from numpy import vectorize, diff
import numpy as np
import re # regular expressions
import os
from scipy import mat, linalg, double

def read_hessian(filename, n_coords): 
    """returns the hessian (array)"""
    dummy = []
    f = open(filename, 'r')
    hessian = zeros((n_coords, n_coords))

    #dummy = f.readline()
    for i in range(n_coords):
        a = f.readline().split()
        for j in range(n_coords):
            hessian[i,j] = float(a[j])
            print hessian[i,j]
    f.close()

    return hessian

def hessian_trans_rot(hessian, cart_coord, nr_normal_modes, n_atoms): 
    """returns the mass weighted hessian (matrix)"""

    trans1 = [1,0,0]
    trans2 = [0,1,0]
    trans3 = [0,0,1]

    trans_rot = zeros((3* n_atoms, 6))

    for atom in range(n_atoms):

        ij = atom*3
        trans_rot[ij, 0] = 1.0
        trans_rot[ij + 1, 1] = 1.0
        trans_rot[ij + 2, 2] = 1.0
        trans_rot[ij, 3] = -1* cart_coord[atom, 1]
        trans_rot[ij + 1, 3] = cart_coord[atom, 0]
        trans_rot[ij + 1, 4] = -1* cart_coord[atom, 2]
        trans_rot[ij + 2, 4] = cart_coord[atom, 1]
        trans_rot[ij, 5] = cart_coord[atom, 2]
        trans_rot[ij + 2, 5] = -1* cart_coord[atom, 0]

        ij = ij + 3

    trans_rot = linalg.qr(mat(trans_rot), mode = 'economic') [0:1]
    trans_rot = -1* mat(trans_rot[0])

    trans_rot_proj = -(trans_rot * (trans_rot.T) - mat(identity(3* n_atoms)))  
    hess_proj = mat((trans_rot_proj * mat(hessian)) * trans_rot_proj )
    return hess_proj	#reffered to as Analytical Hessian in DALTON.OUT
                                    
def read_molecule(filename): 
    """returns coordinates(array), mass(list), num_atoms_list, charge_list, sum(num_atoms_list)"""
    f = open(filename, 'r')
    charge_list = []
    num_atoms_list = []
    atomicmass1 = {'O': 15.9994, 'H': 1.00794}

    coordinates = [] # contains the [x,y,z] coordinates of the input atoms
    mass = []   # contains the corresponding masses of the atoms

    finished = 0

    while (finished == 0): 
        mline = re.search('(?<=ypes\=)\w+',f.readline())

        if mline:
            atomtypes = int(mline.group(0)) #The input file specifies how many different types of atoms are present
            finished = 1
    while(atomtypes > 0):
        mline = f.readline()
        charge = int(re.search('(?<=arge\=)\w+',mline).group(0))
        num_atoms = int(re.search('(?<=toms\=)\w+',mline).group(0))
        charge_list.append(charge)
        num_atoms_list.append(num_atoms)

        for i in range(num_atoms):
            mline = f.readline().split()
            mass.append(atomicmass1[mline[0]])
            coordinates.append(mline[1:])

        atomtypes -= 1

    f.close()

    coordinates = array(coordinates, double)

    return coordinates, mass, num_atoms_list, charge_list, sum(num_atoms_list)
    
def masswt_hessian(num_atoms_list, charge_list): 
    """returns mass (array)"""
    atomicmass = {8.0: 15.994915, 1.0: 1.007825}
    m_e = 1822.8884796 # conversion factor from a.m.u to a.u 

    M = zeros((3*sum(num_atoms_list),3*sum(num_atoms_list)))
    s = 0

    for i in range(len(num_atoms_list)):
        for j in 3*range(num_atoms_list[i]):
            M[s, s] = 1/(sqrt(atomicmass[charge_list[i]]*m_e))
            s += 1
    return M

def read_cubic_force_field(filename, n_coords):
    """returns the cubic force field (3d array)"""
    dummy = []
    cubic_force_field = zeros((n_coords, n_coords, n_coords))
    f = open(filename, 'r')

    for i in range(n_coords):
        dummy = f.readline()
        for j in range(n_coords):
            a = f.readline().split()
            for k in range(n_coords):
                cubic_force_field[i,j,k] = float(a[k])
    f.close()

    return cubic_force_field	

def read_eigenvector(filename, n_atoms):
    
    n_nm = 3*n_atoms - 6
    
    eigenvector = zeros((12, 6))

    f = open(filename, 'r')
    
    for i in range(12):
    	a = f.readline().split()
    	for j in range(6):
    		eigenvector[i,j] = float(a[j])
    
    f.close()
    return eigenvector
    
def fundamental_freq(hessian, num_atoms_list, charge_list, molecule, n_atoms): 
    """returns eigentvalues(array)"""
   
    M_I = masswt_hessian(num_atoms_list, charge_list)
    n_nm = 3 * n_atoms - 6
    linear = 0 

    if (linear):
        n_nm += 1

    hessian_proj = dot(M_I.transpose(), hessian_trans_rot(hessian, molecule, n_nm, n_atoms))
    hessian_proj = dot(hessian_proj, M_I)
    v, La = linalg.eig(hessian_proj)
    
    print "All the eigenvectors"
    print v
        
    v_reduced = v[:n_nm]
    
    v_args = v_reduced.argsort()[::-1]
    v_reduced = sort(array(v_reduced, double))
    v_reduced = v_reduced[::-1]
       
    La = dot(M_I, array(La, double))
    La_reduced =  La[:,:n_nm]
    La_reduced = La_reduced[:,v_args]
    
    freq = sqrt(absolute(v))
    freq = sort(array(freq, double))
    freq = freq[::-1]
    
    closefunc = lambda x: abs(x) < 0.0001
    closevect = vectorize(closefunc)
    reldiff = lambda x, y: x/y
    reldiff = vectorize(reldiff)
    eqsign = lambda x, y: x*y > 0
    eqsign = vectorize(eqsign)

    #print "\n\nIs the sign equal?\n"
    #print eqsign(correct_EVEC, La_reduced)
    #print "\n\nAbsolute error\n"
    #print(absolute(correct_EVEC) - absolute(La_reduced))
    #print "\n\nIs the absolute error less than 0.0001?\n"
    #print(closevect(absolute(correct_EVEC) - absolute(La_reduced))) 
    #print "\n\nRelative error of absolute values\n"
    #print(reldiff(absolute(correct_EVEC),absolute(La_reduced)))
    #print "\n\nRaw La\n"
    
    return v_reduced, La_reduced, freq, La

def to_normal_coordinates():
    """returns normal coordinates (array)"""
    mass_temp = zeros((n_atoms,3)) # Make this with coordeq.len or something, make an if loop
    mass_temp[0, :] = masses[0]
    mass_temp[1, :] = masses[1]
    mass_temp[2, :] = masses[2]
    mass_temp[3, :] = masses[3] 

    qi = (coordinates - coordinates_eq)*sqrt(mass_temp)
    qi = reshape(qi,12)
    norm_coordiates =  multiply(eigvec, qi)

    print coordinates
    print mass_temp
    print masses

    return n_normal_coords

def to_normal_coordinates_3D(cubic_force_field, eigvec, n_atoms):
    """returns normal coordinates (array)"""
    
    n_coords = 3* n_atoms
    n_nm = 3 * n_atoms - 6
    cff_norm = zeros((n_coords, n_coords, n_coords)) 

    for i in range(n_coords):
        for j in range(n_coords):
            for k in range(n_coords):
                temp = 0
                for kp in range(n_coords):
                    temp = temp + cubic_force_field[kp,j,i]* eigvec[kp,k]
                cff_norm[k,j,i]= temp
                
    for i in range(n_coords):
        for j in range(n_coords):
            for k in range(n_coords):
                temp = 0
                for jp in range(n_coords):
                    temp = temp + cff_norm[k,jp,i]* eigvec[jp,j]
                cubic_force_field[k,j,i]= temp
                
    for i in range(n_nm):
        for j in range(n_nm):
            for k in range(n_nm):
                temp = 0
                for ip in range(n_coords):
                    temp = temp + cubic_force_field[k,j,ip]* eigvec[ip,i]
                cff_norm[k,j,i]= temp
    
    reldiff = vectorize(lambda x, y: x/y)   
    return cff_norm, cff_norm[:,:6,:6]	

def to_cartessian_coordinates(normal_coords, n_atoms, eigvec):
    """returns cartessian coordinates (array)"""
    factor = sqrt(1822.8884796) #I DONT KNOW WHY?!
    
    n_nm = 3 * n_atoms - 6
    
    correct_coords = mat([[-0.0001200721,0.0008903518,-0.0015799527]
    ,[0.0002073285,-0.0007928443,-0.0017227255]
    ,[0.0050923237,0.008385961,0.0245363238]
    ,[-0.0064771454,-0.0099334764,0.0278795793]])
    


    # Fortran like implementation:
    #for i in range(n_nm):
    #    cor = 0
    #    for atom in range(n_atoms):
    #        for coor in range(3):
    #            cartessian_coordinates[atom, coor] += normal_coords[i]*eigvec[cor,i]*factor
    #            cor = cor+1			
   
    # Redo:
    
    cartessian_coordinates = np.sum(factor*normal_coords*eigvec, 1)
    
    
    #instead of reshape() this will fail if it cannot be done efficiently:
    cartessian_coordinates.shape = (n_atoms, 3) 
                
    #print "result:"
    #print cartessian_coordinates
    #print "correct:"
    #print correct_coords
    #print "\n\nDIFF: \n"
    #print correct_coords - cartessian_coordinates
    return cartessian_coordinates

def effective_geometry(cff_norm, frequencies, n_atoms):
    factor = sqrt(1822.8884796)
    n_normal_coords = 3 * n_atoms
    n_nm = 3 * n_atoms - 6
    molecular_geometry = zeros((n_nm))
    
    for i in range(n_nm):
        prefix = 1/(4*frequencies[i]**2*factor)
        temp = 0
        for j in range(n_nm): 
            temp = temp + divide(cff_norm[i,j,j], frequencies[j])
        molecular_geometry[i] = -1*temp*prefix 

    return molecular_geometry

def get_3D_property(property_type, pre_property, nm, eig, write_to_file):
    """ Corrects magnetizabilities, rotational g-factor, molecular quadropole moments, and indirect spin-spin coupling"""
    
    m_e = 1822.8884796 # conversion factor from a.m.u to a.u 
    prefactor = 1/(4*m_e)
    corrected_property = zeros((3,3))
    
    for mode in range(nm):
        factor = 1/(sqrt(eig[mode])) # the reduced one
        for i in range(3):
            for j in range(3):
                corrected_property[j,i] += pre_property[mode,j,i]*factor
    
    corrected_property = corrected_property*prefactor
    #nuclear_shield_corrected = nuclear_shield + nuclear_shield_correction
 
    if (write_to_file == True):
		
        filename = os.path.abspath("/home/benedicte/Dropbox/master/The Program/output/" + property_type)
        f = open(filename, "w")
		
        line1 = str(corrected_property[0]).strip('[]')
        line2 = str(corrected_property[1]).strip('[]')
        line3 = str(corrected_property[2]).strip('[]')
		
        f.write(line1 + "\n")
        f.write(line2 + "\n")
        f.write(line3 + "\n")

        f.close()
        
    return corrected_property                 

def get_4D_property(property_type, pre_property, n_nm, n_atom, eig, write_to_file):
    """ Corrects nuclear shieldings, nuclear spin -rotation correction, and nuclear quadropole moments"""
    m_e = 1822.8884796 # conversion factor from a.m.u to a.u 
    prefactor = 1/(4*m_e)
    
    corrected_property = zeros((n_atom,3,3))
    
    for nm in range(n_nm):
        factor = 1/(sqrt(eig[nm])) # the reduced one
        for atom in range(n_atom):
            for i in range(3):
                for j in range(3):
                    corrected_property[atom,j,i] += pre_property[atom,nm,j,i]*factor
    
    corrected_property = corrected_property*prefactor
    #nuclear_shield_corrected = nuclear_shield + nuclear_shield_correction
    
    if (write_to_file == True):
        filename = os.path.abspath("/home/benedico/Dropbox/master/The Program/output/" + property_type)
        f = open(filename, "w")
        
        for atom in range(n_atom):
            line1 = str(corrected_property[atom][0]).strip('[]')
            line2 = str(corrected_property[atom][1]).strip('[]')
            line3 = str(corrected_property[atom][2]).strip('[]')
		
            f.write(line1 + "\n")
            f.write(line2 + "\n")
            f.write(line3 + "\n")
            
            f.write("\n") # Seperates the 2D matrices making up the 3D matrix

        f.close()
 
    return corrected_property                    
            
def get_dipole_moment(dipole_moment, n_nm, eig, pre_dipole_moment, write_to_file):
    """" Calculates and return the dipole moment of a molecule given it 
    is at the effective geomoetry. """
    m_e = 1822.8884796 # conversion factor from a.m.u to a.u 
    prefactor = 1/(4*m_e)
    dipole_moment_diff = zeros((3))
    eig = absolute(eig)
    
    for i in range(n_nm):
        factor = 1/(sqrt(eig[i])) # the reduced one
        dipole_moment_diff[0] += dipole_moment[i, 0]*factor
        dipole_moment_diff[1] += dipole_moment[i, 1]*factor
        dipole_moment_diff[2] += dipole_moment[i, 2]*factor
    
    dipole_moment_diff = dipole_moment_diff * prefactor
    dipole_moment_corrected = add(pre_dipole_moment, dipole_moment_diff)
    
    if (write_to_file == True):
		
        filename = os.path.abspath("/home/benedico/Dropbox/master/The Program/output/Dipole Moment")
        f = open(filename, "w")
        line = str(dipole_moment_corrected).strip('[]')
        f.write(line + "\n")
        f.close()
    
    return dipole_moment_diff, dipole_moment_corrected
    
def get_dipole_moment1(dipole_moment, n_nm, eig, pre_dipole_moment): # Don't remember what I was thinking/ testing with this methos
    """" Calculates and return the dipole moment of a molecule given it 
    is at the effective geomoetry. """
    
    m_e = 1822.8884796 # conversion factor from a.m.u to a.u 
    prefactor = 1/(4*m_e)
    dipole_moment_diff = zeros((3))
    eig = absolute(eig)
    
    factor = 1/(sqrt(eig)) # the reduced one
    
    dipole_moment_diff = sum(factor.dot(dipole_moment), axis=0)
    dipole_moment_diff = dipole_moment_diff * prefactor 
    dipole_moment_corrected = add(pre_dipole_moment, dipole_moment_diff)
    
    
    
    return dipole_moment_diff, dipole_moment_corrected

def get_polarizabilities(property_type, pre_property, n_nm, eig, polar):
    """ Corrects polarizabilities"""
    
    if(polar):
        corrected_property = get_3D_property(property_type, pre_property, n_nm, eig, write_to_file)

    else:
        m_e = 1822.8884796 # conversion factor from a.m.u to a.u 
        prefactor = 1/(4*m_e)
        corrected_property = zeros((3,3))
    
        for nm in range(n_nm):
            factor = 1/(sqrt(eig[nm])) # the reduced one
            for i in range(3):
                for j in range(3):
                    corrected_property[ifreq,j,i] += pre_property[ifreq,nm,j,i]*factor
    
        corrected_property = corrected_property*prefactor
        if (write_to_file == True):
        
			filename = os.path.abspath("/home/benedico/Dropbox/master/The Program/output/" + property_type)
			f = open(filename, "w")
        
        for atom in range(n_atom):
            line1 = str(corrected_property[atom][0]).strip('[]')
            line2 = str(corrected_property[atom][1]).strip('[]')
            line3 = str(corrected_property[atom][2]).strip('[]')
		
            f.write(line1 + "\n")
            f.write(line2 + "\n")
            f.write(line3 + "\n")
            
            f.write("\n") # Seperates the 2D matrices making up the 3D matrix

        f.close()
        
        
        
    return corrected_property
    
def get_optical_rotation(property_type, pre_property1, pre_property2, n_nm, ifreq, eig, write_to_file):
    """ Corrects optical rotations""" 
    corrected_property1 = get_4D_property(property_type, pre_property1, n_nm, ifreq, eig, write_to_file)
    corrected_property2 = get_4D_property(property_type, pre_property2, n_nm, ifreq, eig, write_to_file)
    
    return corrected_property1, corrected_property2

    

                 
        


