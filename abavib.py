"""
Module for computing vibrational molecular properties. All variables 
needed to compute the relevant properties are also calculated in this 
module

The parameters needed in this modeule is:
filename = The filename of where a hessian is saved
filename = the filename of where the MOLECULE.INP is saved
filename = the filename of where the cubic force field is saved
pre_property = the second derivative of the property vibrational 
               averaging is to be conducted one
uncorrected_property = the property before the vibrational averaging
correct_propert = The property after the vibrational averaging, for 
                  testing purposes
"""

from numpy import array, zeros, vstack, dot, identity, sqrt, set_printoptions, compress, reshape, multiply, divide, add, subtract, diag, absolute, sort, argsort, fliplr
from numpy import vectorize, diff
import numpy as np
import re # regular expressions
import os
from scipy import mat, linalg, double
import pydoc

def read_hessian(filename, n_coords): 
    """Reads a hessian from file.
    
    filename: The name of the file the hessian is contained in. 
    n_coords: The number of cartessian coordinates needed to express
              the location of the molecule ie. 3 * number of atoms 
    return: A 2-dimensional np.array containg the hessian
    """
    dummy = []
    f = open(filename, 'r')
    hessian = zeros((n_coords, n_coords))

    #dummy = f.readline()
    for i in range(n_coords):
        a = f.readline().split()
        for j in range(n_coords):
            hessian[i,j] = float(a[j])
    f.close()

    return hessian

def hessian_trans_rot(hessian, cart_coord, nr_normal_modes, n_atoms): 
    """Projects the hessian of the molecule so that it can be used to 
    determine the normal coordinates of the molecule. This projected
    hessian is referred to as the analystical hessian in DALTON
    
    hessian: The hessian of the molecule as an np.array
    nr_normal_modes: The number of normal modes of the molecule as an 
                     int
    n_atoms: The number of atoms the molecule consists of
    return: The projected hessian as a 2 dimensional matrix  				 
    """

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
    """Reads the MOLECULE.INP file generated be DALTON, and extracts the 
    information we need from it.
    
    filename: The name of the MOLECULE.INP file as saved in the input
              directory
    return: The cartessian coordinates of the molecule as a list
            The masses of the atoms as a list
            The number of atoms as a list
            The charge of the atoms as a list
            The number of atoms in the molecule as an int
            The atoms making up the molecule as a list
    """
    
    f = open(filename, 'r')
    charge_list = []
    num_atoms_list = []
    atom_list =[]
    atomicmass1 = {'O': 15.9994, 'H': 1.00794, 'C':12.0107, 'D':2.013553212724,'T':3.0160492, 'F':18.998403}
    
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
            atom_list.append(mline[0])
            mass.append(atomicmass1[mline[0]])
            coordinates.append(mline[1:4])

        atomtypes -= 1

    f.close()
    coordinates = array(coordinates, double)

    return coordinates, mass, num_atoms_list, charge_list, sum(num_atoms_list),  atom_list

def mass_hessian(masses):
    """
    Creates an np.array where the masses of the different atoms in
    the molecule are placed at the diagonal. This is a help function
    used to make a mass weighted hessian. 
    masses: The masses of the atoms in the molecule, can be recieved
            from the read_molecule() function.
    returns: The masses alond the diagonal of a 2 dimensional np.array
    """
    
    masses = array(masses)
    m = zeros(3*len(masses))
    m_e = 1822.8884796 # conversion factor from a.m.u to a.u 
    identity_matrix = identity(3*len(masses))
    
    index = 0
    for i in range(len(masses)):
        m[index] = masses[i]
        index = index +1
        m[index] = masses[i]
        index = index +1
        m[index] = masses[i]
        index = index +1
        
    m = m * m_e
    m = 1/(sqrt(m))
    
    M = identity_matrix*m
    
    return M
        
def masswt_hessian(num_atoms_list, charge_list): 
    """returns mass (array)"""
    atomicmass = {9.0:18.998403, 8.0: 15.994915, 1.0: 1.007825}
    
    deuterium = 2.014102
    tritinum = 3.016049
    
    m_e = 1822.8884796 # conversion factor from a.m.u to a.u 

    M = zeros((3*sum(num_atoms_list),3*sum(num_atoms_list)))
    s = 0

    for i in range(len(num_atoms_list)):
        for j in 3*range(num_atoms_list[i]):
            M[s, s] = 1/(sqrt(atomicmass[charge_list[i]]*m_e))
            s += 1
    return M

def read_cubic_force_field(filename, n_coords):
    """
    Reads the cubic force field calculated by DALTON from file.
    
    filename: The name of the file the cubic force field is contained in 
    n_coords: The number of cartessian coordinates needed to express
              the location of the molecule ie. 3 * number of atoms 
    return: A 3-dimensional np.array containg the cubic force field
    """
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
    
def fundamental_freq(hessian, num_atoms_list, charge_list, coordinates, n_atoms, masses): 
    """Computes the normal coordinates, eigenvalues and fundamental 
    frequencies of the molecule. 
    hessian: The hessian of the molecule as an np.array
    num_atoms_list: A list of the how many of each atoms type there are
    charge list: A list over the charges of the atoms in the molecle
    coordinates: The cartessian coordinates of the molecule as an np.array
    n_atoms: The number of atoms composing the molecule as an int
    masses: The masses of the atoms in a molecule along the diagonal of
            an np.array
    returns: The non-zero eigenvalues of the molecule as an np.array
             The normal coordinates of the molecule, ie. the eigenvectors
             corresponding the non-zero eigevalues as an np.array
             The fundalmental frequencies of the molecule as an np.array
             All the eigenvectos of the molecules, both the ones 
             corresponding the zero and non-zero eigenvalues as an np.array
    """
   
    M_I = mass_hessian(masses)
    n_nm = 3 * n_atoms - 6
    linear = 0 

    if (linear):
        n_nm += 1

    hessian_proj = dot(M_I.transpose(), hessian_trans_rot(hessian, coordinates, n_nm, n_atoms))
    hessian_proj = dot(hessian_proj, M_I)
    v, La = linalg.eig(hessian_proj)

    v_reduced = v[:n_nm]
    
    v_args = v_reduced.argsort()[::-1]
    v_reduced = sort(array(v_reduced, double))
    v_reduced = v_reduced[::-1]
    
    for i in range(v_reduced.size):
        if (v_reduced[i] < 0):
            v_reduced[i] = 1
       
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

    return v_reduced, La_reduced, freq, La

def to_normal_coordinates_3D(cubic_force_field, eigvec, n_atoms):
    """Converts cubic force fields represented by cartessina coordinates
    into cubic force field represented by normal coordinates
    
    cubic_force_field: A 3 dimenesional np.array of the cubic force field
                       represented in cartessian coordinates
    eigvec: The eigenvectors of the molecule corresponging to the non
            zero eigenvalues, can be attained for fundamental_freq() (np.array)
    n_atoms: The number of atoms constituting the molecule as an int
    returns: The cubic force field in normal coordinates (np.array)
    """
        
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
    """Converts normal coordinates into cartessian coordinates.
    
    normal coordinates: The normal coordinates of which are to be converted
    n_atoms: The number of atoms constituting the molecule as an int
    eigvec: The eigenvectors of the molecule corresponging to the non
            zero eigenvalues, can be attained for fundamental_freq() (np.array)
    returns: cartessian coordinates as an np.array.
    """
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

    return cartessian_coordinates

def effective_geometry(cff_norm, frequencies, n_atoms):
    """Computes the effective geometry of a molecule.
    
    cff_norm: The cubic force field of the molecule in normal coordinates
              as an np.array
    frequencies: The fundamental frequencies of the molecule as an np.array
    n_atoms: The number of atoms constituting the molecule as an int
    return: The effective geometry in normal coordinates as an np.arrays
    """
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

def get_3D_property(property_type, pre_property, uncorrected_property, nm, eig):
    """ Calculated the vibrationally averaged corrections for first 
    tensor properties. These properties are: magnetizabilities, 
    rotational g-factor, molecular quadropole moments, and indirect 
    spin-spin coupling, these corrections are then added to the uncorrected
    propery
    property_type: The name of the first tensor property, used when
                   writing to file
    pre_property: The second derivative of the property
    Uncorrected property: The property before the corrections
    nm: Number of normal modes for of the molecule
    eig: The non-zero eigenvalues of the molecules hessian
    returns: The corrections to the property, the corrected property as
             np.arrays
    """

    m_e = 1822.8884796 # conversion factor from a.m.u to a.u 
    prefactor = 1/(4*m_e)
    correction_property = zeros((3,3))
    for mode in range(nm):
        factor = 1/(sqrt(eig[mode])) # the reduced one
        for i in range(3):
            for j in range(3):
                correction_property[j,i] += pre_property[mode,j,i]*factor
    
    correction_property = correction_property*prefactor
    corrected_property = uncorrected_property + correction_property 
        
    return correction_property, corrected_property                 

def get_4D_property(property_type, pre_property, uncorrected_property, n_nm, n_atom, eig):
    """ Calculated the vibrationally averaged corrections for first 
    tensor properties. These properties are: nuclear shieldings, nuclear 
    spin -rotation correction, and nuclear quadropole moments, these 
    corrections are then added to the uncorrected propery
    
    property_type: The name of the first tensor property, used when
                   writing to file
    pre_property: The second derivative of the property
    Uncorrected property: The property before the corrections
    nm: Number of normal modes for of the molecule
    eig: The non-zero eigenvalues of the molecules hessian
    n_atoms: The number of atoms constituting the molecule as an int
    returns: The corrections to the property, the corrected property as 
             np.arrays
    """
    
    m_e = 1822.8884796 # conversion factor from a.m.u to a.u 
    prefactor = 1/(4*m_e)
    
    property_corrections = zeros((n_atom,3,3))
    
    for nm in range(n_nm):
        factor = 1/(sqrt(eig[nm])) # the reduced one
        for atom in range(n_atom):
            for i in range(3):
                for j in range(3):
                    property_corrections[atom,j,i] += pre_property[atom,nm,j,i]*factor
    
    property_corrections = property_corrections*prefactor
    corrected_property = property_corrections + uncorrected_property
 
    return property_corrections, corrected_property                    
            
def get_dipole_moment(dipole_moment, n_nm, eig, pre_dipole_moment):
    """" Calculates the corrections to the dipole moment
    
    dipole moment: The uncorrected dipole moment
    n_nm: The number of normal modes of the molecule
    pre_dipole_moment: The second derivative of the dipole moment
    return: The corrections to the dipole moment, the corrected dipole 
    moment as np.arrays 
    """
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
    
    return dipole_moment_diff, dipole_moment_corrected

def get_polarizabilities(property_type, pre_property, n_nm, eig, polar):
    """ Computes polarizability corrections and adds the corrections
    to the original value of the polarizabilty.
    property_type: The name of the first tensor property, used when
                   writing to file
    pre_property: The second derivative of the property
    Uncorrected property: The property before the corrections
    nm: Number of normal modes for of the molecule
    eig: The non-zero eigenvalues of the molecules hessian
    polar: If the molecule is polar or not, as a bollean.
    returns: The corrections to the property, the corrected property as 
             np.arrays
    """
    
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

    return corrected_property, "POLARI"
    




             
    


