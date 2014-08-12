from Property import Property
import Molecule as mol
import read_input as ri
from numpy import array, zeros, absolute, add, sqrt, transpose, divide, multiply, sum, dot
import numpy as np
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
    
        #self.freq = 0.0185
        
        dipole_hessian_diag = self.get_preproperty_ana("input_" + self.molecule.name + "/"+"dh")[1]
        
        self.uncorrected_property = self.get_uncorrected_property()
        eigenvalues = self.molecule.eigenvalues
        correction_property = zeros((3))
        
        for i in range(self.molecule.number_of_normal_modes):
            factor = 1/(sqrt(eigenvalues[i])) # the reduced one
            correction_property[0] += dipole_hessian_diag[0, i]*factor
            correction_property[1] += dipole_hessian_diag[1, i]*factor
            correction_property[2] += dipole_hessian_diag[2, i]*factor
        
        self.correction_property = correction_property * self.prefactor
        self.corrected_property = add(self.uncorrected_property, self.correction_property)
        self.write_to_file(self.property_name)
        
        print("0th order correction")
        print(self.correction_property)
        
        print("0th order corrected")
        print self.corrected_property
        
        #first_order_correction = self.first_order_precision_eq()
        
        #print("1st order correction")
        #print(first_order_correction)
        #print("first order corrected")
        #self.corrected_property = add(self.corrected_property, first_order_correction)
        #print(self.corrected_property)
        
        quartic_correction = self.quartic_precision()
        
        print("2nd order corrected")
        print(self.corrected_property + quartic_correction)
        
        return self.correction_property, self.corrected_property

    def first_order_precision_eq(self):
        """ Calculates the first order corrections to the dipole moment
        uncorrected_property: The uncorrected dipole moment
        pre_property: The second derivative of the dipole moment
        return: The corrections to the dipole moment, the corrected dipole 
        moment as np.arrays correcting to a second order perurbation 
        of the wavefunction"""
        
        prop_deriv = self.get_gradient("input_" + self.molecule.name + "/"+"dg_eq")
        first_order_correction = zeros((3))
        
        cff_cart = ri.read_cubic_force_field_anal(self.molecule.input_name + 'cubic_force_field', self.molecule.n_coordinates)
        cff_norm = self.molecule.to_normal_coordinates_3D(cff_cart)
        first_order_correction =  zeros((3))
        
        factor = np.sum(divide(cff_norm.diagonal(0,0,1)\
        [:self.molecule.number_of_normal_modes,:self.molecule.number_of_normal_modes],self.freq), axis=1)
        first_a1 = -1.0/(4*sqrt(2)*self.freq**(3.0/2))*factor

        for a in range(3):
            first_order_correction[a] = np.sum((sqrt(2)* prop_deriv[:,a]\
            * first_a1)/sqrt(self.freq), axis=0)
        
        
        #First, more intuitive implementation
        
        #first_order_correction =  np.sum((sqrt(2)*prop_deriv*first_a2)/sqrt(self.freq), axis=0)
        #print(first_order_correction2) 
        
        #for a in range(3):
        #    first_a1 = zeros((self.molecule.number_of_normal_modes))
        #    for i in range(self.molecule.number_of_normal_modes):
        #        for m in range(self.molecule.number_of_normal_modes):
        #            prefix_4 = -1.0/(4*sqrt(2)*self.freq[i]**(3.0/2))
        #            first_a1[i] += prefix_4*(cff_norm[i][m][m]/self.freq[m]) 
        #        first_order_correction[a] += (sqrt(2)*prop_deriv[i][a]*first_a2[i])/sqrt(self.freq[i])
        
        first_order_correction = first_order_correction*self.prefactor
        
        return first_order_correction

    def quartic_precision_eq(self):
        """ Calculates the corrections to the dipole moment
        uncorrected_property: The uncorrected dipole moment
        pre_property: The second derivative of the dipole moment
        return: The corrections to the dipole moment, the corrected dipole 
        moment as np.arrays correcting to a second order perurbation 
        of the wavefunction"""
        
        prop_deriv = self.get_preproperty_ana("input_" + self.molecule.name + "/"+"dh_eq")[0]
        
        cff_cart = ri.read_cubic_force_field_anal(self.molecule.input_name + 'cubic_force_field', self.molecule.n_coordinates)
        cff_norm = self.molecule.to_normal_coordinates_3D(cff_cart)
        cff_norm = cff_norm[:self.number_of_normal_modes,:self.number_of_normal_modes, :self.number_of_normal_modes]
        
        qff_cart = ri.read_quartic_force_field(self.molecule.input_name + 'quartic_force_field', self.molecule.n_coordinates)
        qff_norm = self.molecule.to_normal_coordinates_4D(qff_cart)
        qff_norm = qff_norm[:self.number_of_normal_modes,:self.number_of_normal_modes, :self.number_of_normal_modes, :self.number_of_normal_modes]
        
        quartic_correction = zeros((3))
        
        first_a1_factor = np.sum(divide(cff_norm.diagonal(0,0,1)\
        [:self.number_of_normal_modes,:self.number_of_normal_modes],self.freq), axis=1)
        
        first_a1 = -1.0/(4*sqrt(2)*self.freq**(3.0/2))*first_a1_factor
        first_a3 = (sqrt(3.0)*cff_norm.diagonal(0,0,1).diagonal())/36**(5.0/2)
        
        term_11 = -1.0*first_a1*cff_norm.diagonal(0,0,1).diagonal()/(4*self.freq**(3.0/5)) 
        term_12 = -1.0*first_a1*np.sum(divide(cff_norm.diagonal(0,0,1),(8*self.freq)),axis=1)/self.freq**(3.0/2)
        term_13 = -1.0*first_a3*sqrt(27)*cff_norm.diagonal(0,0,1).diagonal()/(sqrt(32)*self.freq**(5.0/2))
        term_14 = -1*first_a3*sqrt(3)*np.sum(divide(cff_norm.diagonal(0,0,1),(8*sqrt(2)*self.freq)), axis=1)/(self.freq**(3.0/2))
        term_15 = -1*sqrt(2)*qff_norm.diagonal(0,0,1).diagonal(0,0,1).diagonal()/(32.0*self.freq**3) 
        term_16 = sqrt(2)*np.sum(qff_norm.diagonal(0,0,1).diagonal(0,0,1)/(8*self.freq),axis= 1)/self.freq**2.0 

        second_a2 = term_11 + term_12 + term_13 + term_14 + term_15 + term_16     
        
        for a in range(3):
        
            second_b11 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
            second_b31 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
                            
            for i in range(self.molecule.number_of_normal_modes):
                
                quartic_correction[a] += second_a2[i]*sqrt(2.0)*prop_deriv[i][i][a]/(4*self.freq[i]) \
                + (first_a3[i]**2 + first_a1[i]**2 + first_a3[i]*first_a1[i])*prop_deriv[i][i][a]/(3*self.freq[i])\
                -prop_deriv[i][i][a]/(2*self.freq[i])
            
            for i in range(self.molecule.number_of_normal_modes):    
                for j in range(self.molecule.number_of_normal_modes):
                    prefix_1 = 1/(32*self.freq[i]**(3.0/2) * self.freq[j]**0.5*(self.freq[i] + self.freq[j]))
                    second_b11[i][j] += prefix_1*qff_norm[i][i][i][j]
                    
                    prefix_1= sqrt(6.0)/(96*self.freq[i]**(3/2) * self.freq[j]**0.5*(3*self.freq[i] + self.freq[j]))
                    second_b31[i][j] += prefix_1*qff_norm[i][i][i][j]
                    
                    for m in range(self.molecule.number_of_normal_modes):
                        prefix_2 = 1/sqrt(2.0) * self.freq[m]*2*self.freq[i]**(1.0/2)*sqrt(2.0)*self.freq[j]**(1.0/2)*(self.freq[i]+self.freq[j])
                        second_b11[i][j] += prefix_2*qff_norm[i][i][i][j]
                
                    quartic_correction[a] += second_b11[i][j]*prop_deriv[i][j][a]/(4*self.freq[i]**(1.0/2)*self.freq[j]**(1.0/2)) \
                                + second_b31[i][j]*sqrt(2.0)*prop_deriv[i][j][a]/(4*self.freq[j])
        
        quartic_correction = quartic_correction*self.prefactor
        
        
        
            # For loop implementation. 
            #
            #for a in range(3):
            #first_a1 = zeros((self.molecule.number_of_normal_modes))
            #first_a3 = zeros((self.molecule.number_of_normal_modes))
            #second_a2 = zeros((self.molecule.number_of_normal_modes))
            #second_b11 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
            #second_b31 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
                    
            #prefix_12 = zeros((self.molecule.number_of_normal_modes))
            #prefix_14 = zeros((self.molecule.number_of_normal_modes))
            #prefix_16 = zeros((self.molecule.number_of_normal_modes))
                            
            #for i in range(self.molecule.number_of_normal_modes):
            #    first_a3[i] += sqrt(3.0)*cff_norm[i][i][i]/36**(5.0/2)
            #    for m in range(self.molecule.number_of_normal_modes):
            #        prefix_4 = -1.0/(4*sqrt(2)*self.freq[i]**(3.0/2))
            #        first_a1[i] += prefix_4*(cff_norm[i][m][m]/self.freq[m])  
            #    for m in range(self.molecule.number_of_normal_modes):
            #        prefix_11 = -1*first_a1[i]*cff_norm[i][i][i]/(4*self.freq[i]**(3.0/5)) 
            #        prefix_12[i] += -1.0*first_a1[i]*cff_norm[i][m][m]/(8*self.freq[m]*self.freq[i]**(3.0/2))
            #        prefix_13 = -1*first_a3[i]*sqrt(27)*cff_norm[i][i][i]/(sqrt(32)*self.freq[i]**(5.0/2))
            #        prefix_14[i] += -1*first_a3[i]*sqrt(3)*cff_norm[i][m][m]/(8*sqrt(2)*self.freq[m]*self.freq[i]**(3.0/2))
            #        prefix_15 = -1*sqrt(2)*qff_norm[i][i][i][i]/(32.0*self.freq[i]**3)
            #        prefix_16[i] += sqrt(2)*qff_norm[i][i][m][m]/(8*self.freq[m]*self.freq[i]**2.0)
            #                  
            #    second_a2[i] += prefix_11 + prefix_12[i]\
            #            + prefix_13 + prefix_14[i] + prefix_15 + prefix_16[i]
                
            #    quartic_correction[a] += second_a2[i]*sqrt(2.0)*prop_deriv[i][i][a]/(4*self.freq[i]) \
            #    + (first_a3[i]**2 + first_a1[i]**2 + first_a3[i]*first_a1[i])*prop_deriv[i][i][a]/(3*self.freq[i])\
            #    -prop_deriv[i][i][a]/(2*self.freq[i])
            #
            #for i in range(self.molecule.number_of_normal_modes):    
            #    for j in range(self.molecule.number_of_normal_modes):
            #        prefix_1 = 1/(32*self.freq[i]**(3.0/2) * self.freq[j]**0.5*(self.freq[i] + self.freq[j]))
            #        second_b11[i][j] += prefix_1*qff_norm[i][i][i][j]
            #        
            #        prefix_1= sqrt(6.0)/(96*self.freq[i]**(3/2) * self.freq[j]**0.5*(3*self.freq[i] + self.freq[j]))
            #        second_b31[i][j] += prefix_1*qff_norm[i][i][i][j]
            #        
            #        for m in range(self.molecule.number_of_normal_modes):
            #            prefix_2 = 0
            #            prefix_2 += 1/sqrt(2.0) * self.freq[m]*2*self.freq[i]**(1.0/2)*sqrt(2.0)*self.freq[j]**(1.0/2)*(self.freq[i]+self.freq[j])
            #            second_b11[i][j] += prefix_2*qff_norm[i][i][i][j]
            #    
            #        quartic_correction[a] += second_b11[i][j]*prop_deriv[i][j][a]/(4*self.freq[i]**(1.0/2)*self.freq[j]**(1.0/2)) \
            #                    + second_b31[i][j]*sqrt(2.0)*prop_deriv[i][j][a]/(4*self.freq[j])
        
        
        print("2nd order correction")
        print quartic_correction        
        
        return quartic_correction
        
    def quartic_precision(self):
        """" Calculates the corrections to the dipole moment
        uncorrected_property: The uncorrected dipole moment
        pre_property: The second derivative of the dipole moment
        return: The corrections to the dipole moment, the corrected dipole 
        moment as np.arrays correcting to a second order perurbation 
        of the wavefunction"""
    
        prop_deriv = self.get_preproperty_ana("input_" + self.molecule.name + "/"+"dh")[0]
        
        cff_cart = ri.read_cubic_force_field_anal(self.molecule.input_name + 'cubic_force_field', self.molecule.n_coordinates)
        cff_norm = self.molecule.to_normal_coordinates_3D(cff_cart)
        cff_norm = cff_norm[:self.number_of_normal_modes,:self.number_of_normal_modes, :self.number_of_normal_modes]
        
        qff_cart = ri.read_quartic_force_field(self.molecule.input_name + 'quartic_force_field', self.molecule.n_coordinates)
        qff_norm = self.molecule.to_normal_coordinates_4D(qff_cart)
        qff_norm = qff_norm[:self.number_of_normal_modes,:self.number_of_normal_modes, :self.number_of_normal_modes, :self.number_of_normal_modes]
        
        quartic_correction = zeros((3))
        
        first_a3 = (sqrt(3.0)*cff_norm.diagonal(0,0,1).diagonal())/36**(5.0/2)
        term_13 = -1.0*first_a3*sqrt(27)*cff_norm.diagonal(0,0,1).diagonal()/(sqrt(32)*self.freq**(5.0/2))
        term_14 = -1*first_a3*sqrt(3)*np.sum(divide(cff_norm.diagonal(0,0,1),(8*sqrt(2)*self.freq)), axis=1)/(self.freq**(3.0/2))
        term_15 = -1*sqrt(2)*qff_norm.diagonal(0,0,1).diagonal(0,0,1).diagonal()/(32.0*self.freq**3) 
        term_16 = sqrt(2)*np.sum(qff_norm.diagonal(0,0,1).diagonal(0,0,1)/(8*self.freq),axis= 1)/self.freq**2.0 

        second_a2 = term_13 + term_14 + term_15 + term_16     
        
        quartic_correction = zeros((3))
        
        for a in range(3):
            second_b11 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
            second_b31 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
                 
            for i in range(self.molecule.number_of_normal_modes):    
                quartic_correction[a] += second_a2[i]*sqrt(2.0)*prop_deriv[i][i][a]/(4*self.freq[i]) + (first_a3[i]**2)*prop_deriv[i][i][a]/(5*self.freq[i])\
                        -prop_deriv[i][i][a]/(2*self.freq[i])
                for j in range(self.molecule.number_of_normal_modes):
                    prefix_1 = 1/(32*self.freq[i]**(3.0/2) * self.freq[j]**0.5*(self.freq[i] + self.freq[j]))
                    second_b11[i][j] += prefix_1*qff_norm[i][i][i][j]
                    
                    prefix_1= sqrt(6.0)/(96*self.freq[i]**(3/2) * self.freq[j]**0.5*(3*self.freq[i] + self.freq[j]))
                    second_b31[i][j] += prefix_1*qff_norm[i][i][i][j]
                    
                    for m in range(self.molecule.number_of_normal_modes):
                        prefix_2 = 1/sqrt(2.0) * self.freq[m]*2*self.freq[i]**(1.0/2)*sqrt(2.0)*self.freq[j]**(1.0/2)*(self.freq[i]+self.freq[j])
                        second_b11[i][j] += prefix_2*qff_norm[i][i][i][j]
                
                    quartic_correction[a] += second_b11[i][j]*prop_deriv[i][j][a]/(4*self.freq[i]**(1.0/2)*self.freq[j]**(1.0/2)) \
                                + second_b31[i][j]*sqrt(2.0)*prop_deriv[i][j][a]/(4*self.freq[j])
        
        quartic_correction = quartic_correction*self.prefactor
        
        print("2nd order correction")
        print quartic_correction        
        
        return quartic_correction
            
    def get_preproperty(self):
        pre_property = ri.read_2d_input(self.molecule.input_name + "/MOLQUAD", self.molecule.number_of_normal_modes)
        return pre_property

    def get_preproperty_ana(self, filename):
        
        pre_property_cart = ri.read_dipole_hessian(filename, self.molecule.n_coordinates)
        pre_property_full, pre_property_diag = self.to_normal_coordinates_2D(pre_property_cart)
        
        return(pre_property_full, pre_property_diag)

    def get_gradient(self, filename):
        gradient_cart = ri.read_dipole_gradient(filename, self.molecule.n_coordinates)
        gradient_norm = self.to_normal_coordinates_1D(gradient_cart)
        
        return gradient_norm
    
    def get_uncorrected_property(self):
        uncorrected_property = ri.read_DALTON_values_2d(self.molecule.input_name + "MOLQUAD")[0]
        #uncorrected_property = self.molecule.hessian_trans_rot(uncorrected_property, self.molecule.coordinates, self.molecule.number_of_normal_modes, self.molecule.n_atoms)
        return uncorrected_property

    def to_normal_coordinates_1D(self, grad): 
        
        #conversion_factor = 205.07454 # (a.u to Debye)*(a.u to a.m.u)*(Angstrom to bohr) #For testing purposes
        conversion_factor = -1822.8884796  #(a.u to a.m.u)
    
        grad_norm = zeros((self.molecule.n_coordinates,3))
        
        grad_norm[:,0] += dot(grad[:,0],self.molecule.eigenvectors_full)
        grad_norm[:,1] += dot(grad[:,1],self.molecule.eigenvectors_full)
        grad_norm[:,2] += dot(grad[:,2],self.molecule.eigenvectors_full)
        
        grad_norm = grad_norm[:self.molecule.number_of_normal_modes,:]*conversion_factor

        # OpenRSP implementation
        #grad_norm = zeros((self.molecule.number_of_normal_modes,3))
        
        #for ip in range(self.molecule.number_of_normal_modes):
        #    temp = zeros((3))
        #    for i in range(self.molecule.n_coordinates):
        #        for j in range(3):
        #            temp[j] = temp[j] + grad[i,j]*self.molecule.eigenvectors_full[i,ip]
        #    grad_norm[ip,:] = temp
        #grad_norm = grad_norm*conversion_factor
        
        
        return(grad_norm)
        
    def to_normal_coordinates_2D(self, dipole_hessian):
        
        conversion_factor = -1822.8884796  #(a.u to a.m.u)
        n_coordinates = self.molecule.n_coordinates
        number_of_normal_modes = self.molecule.number_of_normal_modes
        self.number_of_normal_modes = self.molecule.number_of_normal_modes
        hess_norm = zeros((n_coordinates,n_coordinates,3))
        
        for i in range(3):
            hess_norm[:,:,i] = dot(dipole_hessian[:,:,i],self.molecule.eigenvectors_full)
            hess_norm[:,:,i] = dot(transpose(hess_norm[:,:,i]),self.molecule.eigenvectors_full)
        
        hess_norm = hess_norm[:number_of_normal_modes,:number_of_normal_modes,:]*conversion_factor
        hess_norm = transpose(hess_norm,(1,0,2))
        hess_diag = hess_norm.diagonal(0,0,1)
        
        
        #OpenRSP implementation
        #hess_norm = zeros((n_coordinates,n_coordinates,3))
        #hess_norm_temp = zeros((n_coordinates, n_coordinates, 3)) 
        #for i in range(n_coordinates):
        #    for ip in range(self.number_of_normal_modes):
        #        temp = zeros((3))
        #        for j in range(n_coordinates):
        #            for k in range(3):
        #                temp[k] = temp[k] + dipole_hessian[i,j,k]*self.molecule.eigenvectors_full[j,ip]
        #        hess_norm_temp[i,ip,:] = temp     
        #for i in range(self.number_of_normal_modes):
        #    for ip in range(self.number_of_normal_modes):
        #        temp = zeros((3))
        #        for j in range(n_coordinates):
        #            for k in range(3):
        #                temp[k] = temp[k] + hess_norm_temp[j,ip,k]*self.molecule.eigenvectors_full[j,i]
        #        hess_norm[i,ip,:] = temp
        #hess_diag = hess_norm.diagonal(0,0,1)*conversion_factor
        
        return(hess_norm, hess_diag) 

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
            
            print(self.correction_property)
            print(self.corrected_property)
            
            #qff = self.get_quartic_force_field()
            #qff_norm = self.molecule.to_normal_coordinates_4D(qff)
        
            #quartic_correction = self.quartic_precision(pre_property, qff_norm)
                
            return self.correction_property, self.corrected_property  

    def quartic_precision(self, prop_deriv, qff_norm):
        """" Calculates the corrections to the dipole moment
        uncorrected_property: The uncorrected dipole moment
        pre_property: The second derivative of the dipole moment
        return: The corrections to the dipole moment, the corrected dipole 
        moment as np.arrays correcting to a second order perurbation 
        of the wavefunction"""
            
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

        for nm in range(self.molecule.number_of_normal_modes):
            factor = 1/(sqrt(eigenvalues[nm])) # the reduced one
            for atm in range(self.molecule.n_atoms):
                for i in range(3):
                    for j in range(3):
                        correction_property[atm,j,i] += pre_property[atm,nm,j,i]*factor
    
        self.correction_property = correction_property*self.prefactor
        self.corrected_property = self.correction_property + self.uncorrected_property
        
        
        print(self.correction_property)
        print("corrected property")
        print(self.corrected_property)        
        
        #self.write_to_file(self.property_name, self.molecule.n_atoms)
        #qff = self.get_quartic_force_field()
        #qff_norm = self.molecule.to_normal_coordinates_4D(qff)
        #quartic_correction = self.quartic_precision(pre_property, qff_norm)
        
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

class Polarizability(Property):
    
    def __init__(self, molecule, property_name):
        Property.__init__(self, molecule, property_name)
        self.molecule = molecule
        self.property_name = property_name

    def __call__(self):
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
        
        correction_property = zeros((3,3))
        pre_property = self.get_preproperty_ana("input_" + self.molecule.name + "/"+"ph")[1]
        uncorrected_property = self.get_uncorrected_property() 
        eigenvalues = self.molecule.eigenvalues

        m_e = 1822.8884796 # conversion factor from a.m.u to a.u 
        prefactor = 1/(4*m_e)
        corrected_property = zeros((3,3))
    
        for mode in range(self.molecule.number_of_normal_modes):
            factor = 1/(sqrt(eigenvalues[mode])) # the reduced one
            for i in range(3):
                for j in range(3):
                    correction_property[j,i] += pre_property[mode,j,i]*factor
        
        
        print("Uncorrected property")
        print(uncorrected_property)
        print("correction_property")
        correction_property = correction_property*prefactor
        corrected_property = uncorrected_property + correction_property 

        print(correction_property)
        
        print("Corrected property")
        print(corrected_property)
        
        #first_order_correction = self.first_order_precision()
        
        #print("1st order correction")
        #print(first_order_correction)
        #print("first order corrected")
        #corrected_property = add(corrected_property, first_order_correction)
        #print(corrected_property)
        
        #correction = self.quartic_precision_eq()
        correction = self.quartic_precision()
        
        
        print("2nd order corrected")
        print(corrected_property + correction)

        return corrected_property, "POLARI" 
        
    def to_normal_coordinates_2D_pol(self, dipole_hessian):
        
        hess_norm_temp = zeros((self.molecule.n_coordinates, self.molecule.n_coordinates, 3,3)) 
        hess_norm = zeros((self.molecule.number_of_normal_modes,self.molecule.number_of_normal_modes,3,3))
        
        for i in range(self.molecule.n_coordinates):
            for ip in range(self.molecule.n_coordinates):
                temp = zeros((3,3))
                for j in range(self.molecule.n_coordinates):
                    for k in range(3):
                        for l in range(3):
                            temp[k,l] = temp[k,l] + dipole_hessian[i,j,k,l]*self.molecule.eigenvectors_full[j,ip]
                hess_norm_temp[i,ip,:,:] = temp
        for i in range(self.molecule.number_of_normal_modes):
            for ip in range(self.molecule.number_of_normal_modes):
                temp = zeros((3,3))
                for j in range(self.molecule.n_coordinates):
                    for k in range(3):
                        for l in range(3):
                            temp[k,l] = temp[k,l] + hess_norm_temp[j,ip,k,l]*self.molecule.eigenvectors_full[j,i]
                hess_norm[i,ip,:,:] = temp
                
        hess_diag = hess_norm.diagonal(0,0,1)
        hess_diag = transpose(hess_diag)* -1822.8884796
        
        return(hess_diag) 

    def get_preproperty(self):
        
        pre_property_cart = ri.read_polari_hessian("input_" + self.molecule.name + "/"+"ph", self.molecule.n_coordinates)
        pre_property = self.to_normal_coordinates_2D_pol(pre_property_cart)
        print("pre-property")
        print(pre_property)
        
        return pre_property
        
    def get_uncorrected_property(self):
        
        molecule_path = self.molecule.input_name \
                        + "POLARI"
        
        uncorrected_property  \
        = ri.read_DALTON_POLARI(molecule_path)[0]
        
        return uncorrected_property
        
    def quartic_precision(self):
        """" Calculates the corrections to the dipole moment
        uncorrected_property: The uncorrected dipole moment
        pre_property: The second derivative of the dipole moment
        return: The corrections to the dipole moment, the corrected dipole 
        moment as np.arrays correcting to a second order perurbation 
        of the wavefunction"""
        
        self.number_of_normal_modes = self.molecule.number_of_normal_modes
        
        prop_deriv = self.get_preproperty_ana("input_" + self.molecule.name + "/"+"ph")[0]

        cff_cart = ri.read_cubic_force_field_anal(self.molecule.input_name + 'cubic_force_field', self.molecule.n_coordinates)
        cff_norm = self.molecule.to_normal_coordinates_3D(cff_cart)
        cff_norm = cff_norm[:self.number_of_normal_modes,:self.number_of_normal_modes, :self.number_of_normal_modes]
        
        qff_cart = ri.read_quartic_force_field(self.molecule.input_name + 'quartic_force_field', self.molecule.n_coordinates)
        qff_norm = self.molecule.to_normal_coordinates_4D(qff_cart)
        qff_norm = qff_norm[:self.number_of_normal_modes,:self.number_of_normal_modes, :self.number_of_normal_modes, :self.number_of_normal_modes]
        

        first_a3 = (sqrt(3.0)*cff_norm.diagonal(0,0,1).diagonal())/36**(5.0/2)
        term_13 = -1.0*first_a3*sqrt(27)*cff_norm.diagonal(0,0,1).diagonal()/(sqrt(32)*self.freq**(5.0/2))
        term_14 = -1*first_a3*sqrt(3)*np.sum(divide(cff_norm.diagonal(0,0,1),(8*sqrt(2)*self.freq)), axis=1)/(self.freq**(3.0/2))
        term_15 = -1*sqrt(2)*qff_norm.diagonal(0,0,1).diagonal(0,0,1).diagonal()/(32.0*self.freq**3) 
        term_16 = sqrt(2)*np.sum(qff_norm.diagonal(0,0,1).diagonal(0,0,1)/(8*self.freq),axis= 1)/self.freq**2.0 

        second_a2 = term_13 + term_14 + term_15 + term_16
            
        quartic_correction = zeros((3,3))

        for a in range(3):
            for b in range(3):
                second_b11 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
                second_b31 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
                
                for i in range(self.molecule.number_of_normal_modes):
                    quartic_correction[a,b] += second_a2[i]*sqrt(2.0)*prop_deriv[i][i][a][b]/(4*self.freq[i]) + first_a3[i]**2 * prop_deriv[i][i][a][b]/(3*self.freq[i])\
                            - prop_deriv[i][i][a][b]*1/(2*self.freq[i])
                     
                for i in range(self.molecule.number_of_normal_modes):    
                    for j in range(self.molecule.number_of_normal_modes):
                        prefix_1 = 1/(32*self.freq[i]**(3.0/2) * self.freq[j]**0.5*(self.freq[i] + self.freq[j]))
                        second_b11[i][j] += prefix_1*qff_norm[i][i][i][j]
                        
                        prefix_1= sqrt(6.0)/(96*self.freq[i]**(3/2) * self.freq[j]**0.5*(3*self.freq[i] + self.freq[j]))
                        second_b31[i][j] += prefix_1*qff_norm[i][i][i][j]
                        
                        for m in range(self.molecule.number_of_normal_modes):
                            prefix_2 = 1/sqrt(2.0) * self.freq[m]*2*self.freq[i]**(1.0/2)*sqrt(2.0)*self.freq[j]**(1.0/2)*(self.freq[i]+self.freq[j])
                            second_b11[i][j] += prefix_2*qff_norm[i][i][i][j]
                    
                        quartic_correction[a,b] += second_b11[i][j]*prop_deriv[i][j][a][b]/(4*self.freq[i]**(1.0/2)*self.freq[j]**(1.0/2)) \
                                    + second_b31[i][j]*sqrt(2.0)*prop_deriv[i][j][a][b]/(4*self.freq[j])
        
        quartic_correction = quartic_correction*self.prefactor
        
        print("2nd order correction")
        print quartic_correction        
        
        return quartic_correction

    def first_order_precision_eq(self):
        """ Calculates the first order corrections to the dipole moment
        uncorrected_property: The uncorrected dipole moment
        pre_property: The second derivative of the dipole moment
        return: The corrections to the dipole moment, the corrected dipole 
        moment as np.arrays correcting to a second order perurbation 
        of the wavefunction"""
        
        prop_deriv = self.get_gradient("input_" + self.molecule.name + "/"+"pg_eq")
        first_order_correction = zeros((3,3))
        
        cff_cart = ri.read_cubic_force_field_anal(self.molecule.input_name + 'cubic_force_field', self.molecule.n_coordinates)
        cff_norm = self.molecule.to_normal_coordinates_3D(cff_cart)
        
        first_order_correction =  zeros((3,3))
        
        factor = np.sum(divide(cff_norm.diagonal(0,0,1)\
        [:self.molecule.number_of_normal_modes,:self.molecule.number_of_normal_modes],self.freq), axis=1)
        first_a1 = -1.0/(4*sqrt(2)*self.freq**(3.0/2))*factor
        
        for a in range(3):
            for b in range(3):
                first_order_correction[a,b] = np.sum((sqrt(2)* prop_deriv[:,a,b]\
                * first_a1)/sqrt(self.freq), axis=0)
            
        #for a in range(3):
        #    for b in range(3):
        #        first_a1 = zeros((self.molecule.number_of_normal_modes))
        #        for i in range(self.molecule.number_of_normal_modes):
        #            for m in range(self.molecule.number_of_normal_modes):
        #                prefix_4 = -1.0/(4*sqrt(2)*self.freq[i]**(3.0/2))
        #                first_a1[i] += prefix_4*(cff_norm[i][m][m]/self.freq[m]) 
        #            first_order_correction[a,b] += (sqrt(2)*prop_deriv[i,a,b]*first_a1[i])/sqrt(self.freq[i])
        
        first_order_correction = first_order_correction*self.prefactor

        return first_order_correction

    def quartic_precision_eq(self):
        """" Calculates the corrections to the dipole moment
        uncorrected_property: The uncorrected dipole moment
        pre_property: The second derivative of the dipole moment
        return: The corrections to the dipole moment, the corrected dipole 
        moment as np.arrays correcting to a second order perurbation 
        of the wavefunction"""
        
        self.number_of_normal_modes = self.molecule.number_of_normal_modes
        
        prop_deriv = self.get_preproperty_ana("input_" + self.molecule.name + "/"+"ph_eq")[0]

        cff_cart = ri.read_cubic_force_field_anal(self.molecule.input_name + 'cubic_force_field', self.molecule.n_coordinates)
        cff_norm = self.molecule.to_normal_coordinates_3D(cff_cart)
        cff_norm = cff_norm[:self.number_of_normal_modes,:self.number_of_normal_modes, :self.number_of_normal_modes]
        
        qff_cart = ri.read_quartic_force_field(self.molecule.input_name + 'quartic_force_field', self.molecule.n_coordinates)
        qff_norm = self.molecule.to_normal_coordinates_4D(qff_cart)
        qff_norm = qff_norm[:self.number_of_normal_modes,:self.number_of_normal_modes, :self.number_of_normal_modes, :self.number_of_normal_modes]
        
        first_a1_factor = np.sum(divide(cff_norm.diagonal(0,0,1)\
        [:self.number_of_normal_modes,:self.number_of_normal_modes],self.freq), axis=1)
        
        first_a1 = -1.0/(4*sqrt(2)*self.freq**(3.0/2))*first_a1_factor
        first_a3 = (sqrt(3.0)*cff_norm.diagonal(0,0,1).diagonal())/36**(5.0/2)
        
        term_11 = -1.0*first_a1*cff_norm.diagonal(0,0,1).diagonal()/(4*self.freq**(3.0/5)) 
        term_12 = -1.0*first_a1*np.sum(divide(cff_norm.diagonal(0,0,1),(8*self.freq)),axis=1)/self.freq**(3.0/2)
        term_13 = -1.0*first_a3*sqrt(27)*cff_norm.diagonal(0,0,1).diagonal()/(sqrt(32)*self.freq**(5.0/2))
        term_14 = -1*first_a3*sqrt(3)*np.sum(divide(cff_norm.diagonal(0,0,1),(8*sqrt(2)*self.freq)), axis=1)/(self.freq**(3.0/2))
        term_15 = -1*sqrt(2)*qff_norm.diagonal(0,0,1).diagonal(0,0,1).diagonal()/(32.0*self.freq**3) 
        term_16 = sqrt(2)*np.sum(qff_norm.diagonal(0,0,1).diagonal(0,0,1)/(8*self.freq),axis= 1)/self.freq**2.0 

        second_a2 = term_11 + term_12 + term_13 + term_14 + term_15 + term_16      
            
        quartic_correction = zeros((3,3))
        
        for a in range(3):
            for b in range(3):
                second_b11 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
                second_b31 = zeros((self.molecule.number_of_normal_modes, self.molecule.number_of_normal_modes))
                             
                for i in range(self.molecule.number_of_normal_modes):
                    quartic_correction[a][b] += second_a2[i]*sqrt(2.0)*prop_deriv[i][i][a][b]/(4*self.freq[i]) \
                        + (first_a3[i]**2 + first_a1[i]**2 + first_a3[i]*first_a1[i])*prop_deriv[i][i][a][b]/(3*self.freq[i])- prop_deriv[i][i][a][b]/(2*self.freq[i])    
                    for j in range(self.molecule.number_of_normal_modes):
                        prefix_1 = 1/(32*self.freq[i]**(3.0/2) * self.freq[j]**0.5*(self.freq[i] + self.freq[j]))
                        second_b11[i][j] += prefix_1*qff_norm[i][i][i][j]
                        
                        prefix_1= sqrt(6.0)/(96*self.freq[i]**(3/2) * self.freq[j]**0.5*(3*self.freq[i] + self.freq[j]))
                        second_b31[i][j] += prefix_1*qff_norm[i][i][i][j]
                        
                        for m in range(self.molecule.number_of_normal_modes):
                            prefix_2 = 1/sqrt(2.0) * self.freq[m]*2*self.freq[i]**(1.0/2)*sqrt(2.0)*self.freq[j]**(1.0/2)*(self.freq[i]+self.freq[j])
                            second_b11[i][j] += prefix_2*qff_norm[i][i][i][j]
                    
                        quartic_correction[a,b] += second_b11[i][j]*prop_deriv[i][j][a][b]/(4*self.freq[i]**(1.0/2)*self.freq[j]**(1.0/2)) \
                                    + second_b31[i][j]*sqrt(2.0)*prop_deriv[i][j][a][b]/(4*self.freq[j])
        
        quartic_correction = quartic_correction*self.prefactor
        
        print("2nd order correction")
        print quartic_correction        
        
        return quartic_correction

    def get_preproperty_ana(self, filename):
        
        pre_property_cart = ri.read_polari_hessian(filename, self.molecule.n_coordinates)
        pre_property_full, pre_property_diag = self.to_normal_coordinates_2D_pol(pre_property_cart)
        
        return(pre_property_full, pre_property_diag)

    def get_gradient(self, filename):
        gradient_cart = ri.read_polari_gradient(filename, self.molecule.n_coordinates)
        gradient_norm = self.to_normal_coordinates_1D_pol(gradient_cart)
        
        return gradient_norm
        
    def to_normal_coordinates_1D_pol(self, grad): 
        
        conversion_factor = -1822.8884796 # (a.u to Debye)*(a.u to a.m.u)*(Angstrom to bohr) 
        grad_norm = zeros((self.molecule.number_of_normal_modes,3,3))
        
        for ip in range(self.molecule.number_of_normal_modes):
            temp = zeros((3,3))
            for i in range(self.molecule.n_coordinates):
                for j in range(3):
                    for k in range(3):
                        temp[j,k] = temp[j,k] + grad[i,j,k]*self.molecule.eigenvectors_full[i,ip]
            grad_norm[ip,:,:] = temp
        grad_norm = grad_norm*conversion_factor
        
        return(grad_norm) 
        
    def to_normal_coordinates_2D_pol(self, dipole_hessian):
        
        hess_norm_temp = zeros((self.molecule.n_coordinates, self.molecule.n_coordinates, 3,3)) 
        hess_norm = zeros((self.molecule.number_of_normal_modes,self.molecule.number_of_normal_modes,3,3))
        
        for i in range(self.molecule.n_coordinates):
            for ip in range(self.molecule.n_coordinates):
                temp = zeros((3,3))
                for j in range(self.molecule.n_coordinates):
                    for k in range(3):
                        for l in range(3):
                            temp[k,l] = temp[k,l] + dipole_hessian[i,j,k,l]*self.molecule.eigenvectors_full[j,ip]
                hess_norm_temp[i,ip,:,:] = temp
        for i in range(self.molecule.number_of_normal_modes):
            for ip in range(self.molecule.number_of_normal_modes):
                temp = zeros((3,3))
                for j in range(self.molecule.n_coordinates):
                    for k in range(3):
                        for l in range(3):
                            temp[k,l] = temp[k,l] + hess_norm_temp[j,ip,k,l]*self.molecule.eigenvectors_full[j,i]
                hess_norm[i,ip,:,:] = temp
        hess_norm = hess_norm *-1824.8884796        
        hess_diag = hess_norm.diagonal(0,0,1)
        hess_diag = transpose(hess_diag)
       
        return(hess_norm, hess_diag)
