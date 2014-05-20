from Property import Property
import Molecule as mol
import read_input as ri
from numpy import array, zeros, absolute, add, sqrt
import pydoc

class Property_1_Tensor(Property):
    """" Calculates the corrections to the dipole moment
    uncorrected_property: The uncorrected dipole moment
    n_nm: The number of normal modes of the molecule
    pre_property: The second derivative of the dipole moment
    return: The corrections to the dipole moment, the corrected dipole 
    moment as np.arrays"""  
    def __init__(self, molecule, property_name):
        Property.__init__(self, molecule, property_name)
        self.molecule = molecule
        self.property_name = property_name
    
    def __call__(self): 
        
        pre_property = self.get_preproperty()
        self.freq = 0.0185
        self.uncorrected_property = self.get_uncorrected_property()
        eigenvalues = self.molecule.eigenvalues
        correction_property = zeros((3))
        #eigenvalues = absolute(eigenvalues)
        eigenvalues = array([0.00034225]) 
        
        for i in range(self.molecule.number_of_normal_modes):
            factor = 1/(sqrt(eigenvalues[i])) # the reduced one
            correction_property[0] += pre_property[i, 0]*factor
            correction_property[1] += pre_property[i, 1]*factor
            correction_property[2] += pre_property[i, 2]*factor
        
        self.correction_property = correction_property * self.prefactor
        self.corrected_property = add(self.uncorrected_property, self.correction_property)
        self.write_to_file(self.property_name)
        
        print("0th order correction")
        print self.corrected_property
        
        qff = self.get_quartic_force_field()
        qff_norm = self.molecule.to_normal_coordinates_4D(qff)
        
        quartic_correction = self.quartic_precision(pre_property, qff_norm)
        
        second_corrrection = self.corrected_property + quartic_correction 
        
        print("Second order Corrected")
        print(second_corrrection)
        
        
        return self.correction_property, self.corrected_property
    
    def quartic_precision(self, prop_deriv, qff_norm): # Not even close to complete
        """" Calculates the corrections to the dipole moment
        uncorrected_property: The uncorrected dipole moment
        pre_property: The second derivative of the dipole moment
        return: The corrections to the dipole moment, the corrected dipole 
        moment as np.arrays correcting to a second order perurbation 
        of the wavefunction"""
        
        prop_grad = [0]
        
        cff_norm = array([[[0.00002662753]]])
        self.freq = array([0.0185])
        
        self.freq =array([0.00034225])
        qff_norm = array([[[[10.345714]]]])
        
        first_a3 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
        second_a2 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
        second_b11 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
        second_b31 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
        
        quartic_correction = zeros((3))
        
        for a in range(3):
            for i in range(self.molecule.number_of_normal_modes):
                first_a3[i] = sqrt(3.0)*cff_norm[i][i][i]/36**(5.0/2)
                
                prefix_1 = 3*sqrt(3)*first_a3[i]/(8*sqrt(2))*self.freq[i]**(5.0/2)
                prefix_2 = sqrt(2)/(32.0*self.freq[i]**3)
                second_a2 += prefix_1*cff_norm[i][i][i] - prefix_2*qff_norm[i][i][i][i]
               
                for m in range(self.molecule.number_of_normal_modes):
                    prefix_3 = sqrt(2.0)/(8*self.freq[m]*self.freq[i]**2)
                    second_a2 += prefix_3*qff_norm[i][i][m][m]
                
                quartic_correction[a]= second_a2[i]*sqrt(2.0)*prop_deriv[i][a]/(4*self.freq[i]) + first_a3[i]**2 * prop_deriv[i][a]/(3*self.freq[i])\
                        - prop_grad[i]*1/(2*self.freq[i])
                 
            for i in range(self.molecule.number_of_normal_modes):    
                for j in range(self.molecule.number_of_normal_modes):
                    prefix_1= 1/(32*self.freq[i]**(3.0/2) * self.freq[j]**0.5*(self.freq[i] + self.freq[j]))
                    second_b11[i][j] += prefix_1*qff_norm[i][i][i][j]
                    
                    prefix_1= sqrt(6.0)/(96*self.freq[i]**(3/2) * self.freq[j]**0.5*(3*self.freq[i] + self.freq[j]))
                    second_b31[i][j] += prefix_1*qff_norm[i][i][i][j]
                    
                    for m in range(self.molecule.number_of_normal_modes):
                        prefix_2 = 1/sqrt(2.0) * self.freq[m]*2*self.freq[i]**(1.0/2)*sqrt(2.0)*self.freq[j]**(1.0/2)*(self.freq[i]+self.freq[j])
                        second_b11[i][j] += prefix_2*qff_norm[i][i][i][j]
                
                    quartic_correction[a] = second_b11[i][j]*prop_deriv[i][a]/(4*self.freq[i]**(1.0/2)*self.freq[j]**(1.0/2)) \
                                + second_b31[i][j]*sqrt(2.0)*prop_deriv[i][a]/(4*self.freq[j])
        
        quartic_correction = quartic_correction*self.prefactor
        
        print("2nd order correction")
        print quartic_correction        
        return quartic_correction
            
    def get_preproperty(self):
        pre_property = ri.read_2d_input(self.molecule.input_name + "/SHIELD", self.molecule.number_of_normal_modes)
        return pre_property
        
    def get_prop_grad(self):
        prop_grad = 5
        return prop_grad
    
    def get_uncorrected_property(self):
        uncorrected_property = ri.read_DALTON_values_2d(self.molecule.input_name + "SHIELD")[0]
        #uncorrected_property = self.molecule.hessian_trans_rot(uncorrected_property, self.molecule.coordinates, self.molecule.number_of_normal_modes, self.molecule.n_atoms)
        return uncorrected_property
            
class Property_2_Tensor(Property):
    """" Calculates the corrections to the g-factors, nuclear spin-rotations,
    ,molecular quadropole moments, and spin-spin couplings
    uncorrected_property: The uncorrected property
    n_nm: The number of normal modes of the molecule
    pre_property: The second derivative of the property
    return: The corrections to the property, the corrected property 
    as np.arrays""" 
    
    def __init__(self, molecule, property_name):
        Property.__init__(self, molecule, property_name)
        self.molecule = molecule
        self.property_name = property_name
        self.name_dic = {"Magnetizability":"MAGNET", "g-factor": "GFACTOR",\
         "Molecular quadropole moment": "MOLQUAD", "Optical rotation": "OPTROT"}
         
        self.read_dic = {"Magnetizability":ri.read_MAGNET, "g-factor": ri.read_GFACTOR,\
         "Molecular quadropole moment": ri.read_MOLQUAD, "Optical rotation": ri.read_OPTROT}
         
        self.read_DALTON_dic = {"Magnetizability":ri.read_DALTON_MAGNET, "g-factor": ri.read_DALTON_GFACTOR,\
         "Molecular quadropole moment": ri.read_DALTON_MOLQUAD, "Optical rotation":ri.read_DALTON_OPTROT}
        
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
        if(self.molecule.name == "fluoromethane"):
            pre_property1, pre_property2, pre_property3 = self.get_preproperty()
            self.uncorrected_property = self.get_uncorrected_property() 
            self.uncorrected_property1 = self.uncorrected_property[0]
            self.uncorrected_property2 = self.uncorrected_property[1]
            self.uncorrected_property3 = self.uncorrected_property[2]
            
            eigenvalues = self.molecule.eigenvalues
            
            correction_property1 = zeros((3,3))
            correction_property2 = zeros((3,3))
            correction_property3 = zeros((3,3))
            
            for mode in range(self.molecule.number_of_normal_modes):
                factor = 1/(sqrt(eigenvalues[mode])) # the reduced one
                for i in range(3):
                    for j in range(3):
                        correction_property1[j,i] += pre_property1[mode,j,i]*factor
                        
            self.correction_property1 = correction_property1*self.prefactor
            self.corrected_property1 = self.uncorrected_property1 + self.correction_property1
            
            for mode in range(self.molecule.number_of_normal_modes):
                factor = 1/(sqrt(eigenvalues[mode])) # the reduced one
                for i in range(3):
                    for j in range(3):
                        correction_property2[j,i] += pre_property2[mode,j,i]*factor

            self.correction_property2 = correction_property2*self.prefactor
            self.corrected_property2 = self.uncorrected_property2 + self.correction_property2
            
            for mode in range(self.molecule.number_of_normal_modes):
                factor = 1/(sqrt(eigenvalues[mode])) # the reduced one
                for i in range(3):
                    for j in range(3):
                        correction_property3[j,i] += pre_property3[mode,j,i]*factor
       
            self.correction_property3 = correction_property3*self.prefactor
            self.corrected_property3 = self.uncorrected_property3 + self.correction_property3
        
            self.write_to_file(self.property_name)
                
            return 
        
        else:
            pre_property = self.get_preproperty()
            self.uncorrected_property = self.get_uncorrected_property() 
            eigenvalues = self.molecule.eigenvalues
            correction_property = zeros((3,3))
            
            for mode in range(self.molecule.number_of_normal_modes):
                factor = 1/(sqrt(eigenvalues[mode])) # the reduced one
                for i in range(3):
                    for j in range(3):
                        correction_property[j,i] += pre_property[mode,j,i]*factor
            
            self.correction_property = correction_property*self.prefactor
            self.corrected_property = self.uncorrected_property + self.correction_property 
        
            self.write_to_file(self.property_name)
            
            qff = self.get_quartic_force_field()
            qff_norm = self.molecule.to_normal_coordinates_4D(qff)
        
            quartic_correction = self.quartic_precision(pre_property, qff_norm)
                
            return self.correction_property, self.corrected_property  

    def quartic_precision(self, prop_deriv, qff_norm):
        """" Calculates the corrections to the dipole moment
        uncorrected_property: The uncorrected dipole moment
        pre_property: The second derivative of the dipole moment
        return: The corrections to the dipole moment, the corrected dipole 
        moment as np.arrays correcting to a second order perurbation 
        of the wavefunction"""
       
        prop_grad = [0] 
        cff_norm = array([[[0.00002662753]]])
        self.freq = array([0.0185])
            
        self.freq =array([0.00034225])
            
        first_a3 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
        second_a2 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
        second_b11 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
        second_b31 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
            
        quartic_correction = zeros((3,3))
        
        for i in range(self.molecule.number_of_normal_modes):
            for b in range(3):
                for a in range(3):
                    first_a3[i] = sqrt(3.0)*cff_norm[i][i][i]/36**(5.0/2)
                    
                    prefix_1 = 3*sqrt(3)*first_a3[i]/(8*sqrt(2))*self.freq[i]**(5.0/2)
                    prefix_2 = sqrt(2)/(32.0*self.freq[i]**3)
                    second_a2 += prefix_1*cff_norm[i][i][i] - prefix_2*qff_norm[i][i][i][i]
                   
                    for m in range(self.molecule.number_of_normal_modes):
                        prefix_3 = sqrt(2.0)/(8*self.freq[m]*self.freq[i]**2)
                        second_a2 += prefix_3*qff_norm[i][i][m][m]
                    
                    quartic_correction[a][b]= second_a2[i]*sqrt(2.0)*prop_deriv[i][a][b]/(4*self.freq[i]) + first_a3[i]**2 * prop_deriv[i][a][b]/(3*self.freq[i])\
                            - prop_grad[i]*1/(2*self.freq[i])
                     
                for i in range(self.molecule.number_of_normal_modes):    
                    for j in range(self.molecule.number_of_normal_modes):
                        prefix_1= 1/(32*self.freq[i]**(3.0/2) * self.freq[j]**0.5*(self.freq[i] + self.freq[j]))
                        second_b11[i][j] += prefix_1*qff_norm[i][i][i][j]
                        
                        prefix_1= sqrt(6.0)/(96*self.freq[i]**(3/2) * self.freq[j]**0.5*(3*self.freq[i] + self.freq[j]))
                        second_b31[i][j] += prefix_1*qff_norm[i][i][i][j]
                        
                        for m in range(self.molecule.number_of_normal_modes):
                            prefix_2 = 1/sqrt(2.0) * self.freq[m]*2*self.freq[i]**(1.0/2)*sqrt(2.0)*self.freq[j]**(1.0/2)*(self.freq[i]+self.freq[j])
                            second_b11[i][j] += prefix_2*qff_norm[i][i][i][j]
                    
                    quartic_correction[a][b] = second_b11[i][j]*prop_deriv[i][a][b]/(4*self.freq[i]**(1.0/2)*self.freq[j]**(1.0/2)) \
                                + second_b31[i][j]*sqrt(2.0)*prop_deriv[i][a][b]/(4*self.freq[j])
        
        quartic_correction = quartic_correction*self.prefactor
        print("2nd order correction")
        print quartic_correction        
        return quartic_correction
                
    def get_preproperty(self):
        read_property = self.read_dic[self.property_name]
        molecule_path = self.molecule.input_name \
                        + self.name_dic[self.property_name]
        
        pre_property  \
        = read_property(molecule_path, self.molecule.number_of_normal_modes)
        
        if(self.molecule.input_name == "fluoromethane"): # Switch this with property name
            pre_property1, pre_property2, pre_property3   \
            = read_property(molecule_path, self.molecule.number_of_normal_modes)
            return  pre_property1, pre_property2, pre_property3
        
        else:
            pre_property  \
            = read_property(molecule_path, self.molecule.number_of_normal_modes)
            return pre_property
        
    def get_uncorrected_property(self):
        
        read_property = self.read_DALTON_dic[self.property_name]
        molecule_path = self.molecule.input_name \
                        + self.name_dic[self.property_name]
        
        uncorrected_property  \
        = read_property(molecule_path)[0]
        
        return uncorrected_property

class Property_3_Tensor(Property):
    """" Calculates the corrections to the nuclear shieldings,
    nuclear spin correction, nuclear quadropole moment, and optical rotation 
    uncorrected_property: The uncorrected property
    n_nm: The number of normal modes of the molecule
    pre_property: The second derivative of the property
    return: The corrections to the property, the corrected property 
    as np.arrays"""
    
    def __init__(self, molecule, property_name):
        Property.__init__(self, molecule, property_name)
        self.molecule = molecule
        self.property_name = property_name

        self.name_dic = {"Nuclear spin-rotation":"SPINROT", "Nuclear shielding": "SHIELD",\
         "Nuclear quadropole moment": "NUCQUAD", "Optical rotation": "OPTROT"}
         
        self.read_dic = {"Nuclear spin-rotation":ri.read_SPINROT, "Nuclear shielding": ri.read_SHIELD,\
         "Nuclear quadropole moment": ri.read_NUCQUAD, "Optical rotation": ri.read_OPTROT}
         
        self.read_DALTON_dic = {"Nuclear spin-rotation":ri.read_DALTON_SPINROT, "Nuclear shielding": ri.read_DALTON_SHIELD,\
         "Nuclear quadropole moment": ri.read_DALTON_NUCQUAD, "Optical rotation": ri.read_DALTON_OPTROT}
    
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
        
        
        correction_property = zeros((self.molecule.n_atoms,3,3))
        pre_property = self.get_preproperty()[0]
        self.uncorrected_property = self.get_uncorrected_property() 
        eigenvalues = self.molecule.eigenvalues

        #for nm in range(self.molecule.number_of_normal_modes):
            #factor = 1/(sqrt(eigenvalues[nm])) # the reduced one
        #    for atm in range(self.molecule.n_atoms):
        #        for i in range(3):
        #            for j in range(3):
        #                correction_property[atm,j,i] += pre_property[atm,nm,j,i]*factor
    
        #self.correction_property = correction_property*self.prefactor
        #self.corrected_property = self.correction_property + self.uncorrected_property
        
        #self.write_to_file(self.property_name, self.molecule.n_atoms)
        
        qff = self.get_quartic_force_field()
        qff_norm = self.molecule.to_normal_coordinates_4D(qff)
        
        quartic_correction = self.quartic_precision(pre_property, qff_norm)
        
        return 0
        #return self.correction_property, self.corrected_property 
        
    def quartic_precision(self, prop_deriv, qff_norm):
        """" Calculates the corrections to the dipole moment
        uncorrected_property: The uncorrected dipole moment
        pre_property: The second derivative of the dipole moment
        return: The corrections to the dipole moment, the corrected dipole 
        moment as np.arrays correcting to a second order perurbation 
        of the wavefunction"""
        
        prop_grad = [0] 
        cff_norm = array([[[0.00002662753]]])
        self.freq = array([0.0185])
            
        self.freq =array([0.00034225])
            
        first_a3 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
        second_a2 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
        second_b11 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
        second_b31 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
            
        quartic_correction = zeros((self.molecule.n_atoms, 3,3))

        for i in range(self.molecule.number_of_normal_modes):
            for atom in range(self.molecule.n_atoms):
                for b in range(3):
                    for a in range(3):
                        first_a3[i] = sqrt(3.0)*cff_norm[i][i][i]/36**(5.0/2)
                        
                        prefix_1 = 3*sqrt(3)*first_a3[i]/(8*sqrt(2))*self.freq[i]**(5.0/2)
                        prefix_2 = sqrt(2)/(32.0*self.freq[i]**3)
                        second_a2 += prefix_1*cff_norm[i][i][i] - prefix_2*qff_norm[i][i][i][i]
                       
                        for m in range(self.molecule.number_of_normal_modes):
                            prefix_3 = sqrt(2.0)/(8*self.freq[m]*self.freq[i]**2)
                            second_a2 += prefix_3*qff_norm[i][i][m][m]
                        
                        quartic_correction[atom][a][b]= second_a2[i]*sqrt(2.0)*prop_deriv[atom][i][a][b]/(4*self.freq[i]) \
                                + first_a3[i]**2 * prop_deriv[atom][i][a][b]/(3*self.freq[i])\
                                - prop_grad[i]*1/(2*self.freq[i])
                         
                    for i in range(self.molecule.number_of_normal_modes):    
                        for j in range(self.molecule.number_of_normal_modes):
                            prefix_1= 1/(32*self.freq[i]**(3.0/2) * self.freq[j]**0.5*(self.freq[i] + self.freq[j]))
                            second_b11[i][j] += prefix_1*qff_norm[i][i][i][j]
                            
                            prefix_1= sqrt(6.0)/(96*self.freq[i]**(3/2) * self.freq[j]**0.5*(3*self.freq[i] + self.freq[j]))
                            second_b31[i][j] += prefix_1*qff_norm[i][i][i][j]
                            
                            for m in range(self.molecule.number_of_normal_modes):
                                prefix_2 = 1/sqrt(2.0) * self.freq[m]*2*self.freq[i]**(1.0/2)*sqrt(2.0)*self.freq[j]**(1.0/2)*(self.freq[i]+self.freq[j])
                                second_b11[i][j] += prefix_2*qff_norm[i][i][i][j]
                        
                            quartic_correction[atom][a][b] = second_b11[i][j]*prop_deriv[atom][i][a][b]/(4*self.freq[i]**(1.0/2)*self.freq[j]**(1.0/2)) \
                                        + second_b31[i][j]*sqrt(2.0)*prop_deriv[atom][i][a][b]/(4*self.freq[j])
        
        quartic_correction = quartic_correction*self.prefactor
        print("2nd order correction")
        print quartic_correction        
        return quartic_correction
                    
    def get_preproperty(self):
        read_property = self.read_dic[self.property_name]
        molecule_path = self.molecule.input_name \
                        + self.name_dic[self.property_name]
        
        pre_property  \
        = read_property(molecule_path, self.molecule.n_atoms, self.molecule.number_of_normal_modes)
        
        return pre_property
        
    def get_uncorrected_property(self):
        
        read_property = self.read_DALTON_dic[self.property_name]
        molecule_path = self.molecule.input_name \
                        + self.name_dic[self.property_name]
        
        uncorrected_property  \
        = read_property(molecule_path, self.molecule.n_atoms)[0]
        
        return uncorrected_property

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
