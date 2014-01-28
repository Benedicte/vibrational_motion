from Property import Property
import Molecule as mol
import read_input as ri
from numpy import array, zeros, absolute, add, sqrt
import pydoc

class Property_1_Tensor(Property):  
    def __init__(self, molecule, property_name):
        Property.__init__(self, molecule)
        self.molecule = molecule
        self.property_name = property_name
    
    def __call__(self): 
        """" Calculates the corrections to the dipole moment
        uncorrected_property: The uncorrected dipole moment
        n_nm: The number of normal modes of the molecule
        pre_property: The second derivative of the dipole moment
        return: The corrections to the dipole moment, the corrected dipole 
        moment as np.arrays"""
        
        pre_property = self.get_preproperty()
        uncorrected_property = self.get_uncorrected_property()
        eigenvalues = self.molecule.eigenvalues
        correction_property = zeros((3))
        eigenvalues = absolute(eigenvalues)
        
        for i in range(self.molecule.number_of_normal_modes):
            factor = 1/(sqrt(eigenvalues[i])) # the reduced one
            correction_property[0] += pre_property[i, 0]*factor
            correction_property[1] += pre_property[i, 1]*factor
            correction_property[2] += pre_property[i, 2]*factor
        
        correction_property = correction_property * self.prefactor
        corrected_property = add(uncorrected_property, correction_property)
        self.write_to_file(self.property_name, corrected_property)
        return correction_property, corrected_property
        
    def get_preproperty(self):
        pre_property = ri.read_2d_input(self.molecule.input_name + "/SHIELD", self.molecule.number_of_normal_modes)
        return pre_property
        
        
    def get_uncorrected_property(self):
        uncorrected_property = ri.read_DALTON_values_2d(self.molecule.input_name + "SHIELD")[0]
        return uncorrected_property

class Property_2_Tensor(Property):  
    def __call__(self):
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
        correction_property = zeros((3,3))
        for mode in range(nm):
            factor = 1/(sqrt(eig[mode])) # the reduced one
            for i in range(3):
                for j in range(3):
                    correction_property[j,i] += pre_property[mode,j,i]*factor
        
        correction_property = correction_property*prefactor
        corrected_property = uncorrected_property + correction_property 
            
        return correction_property, corrected_property  

class Property_3_Tensor(Property):
    def __call__(self):
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
                 np.arrays"""
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
       
class Polarizability:

    def __call__():
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
                 np.arrays"""
    
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
