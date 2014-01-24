class Property_1_Tensor(Property):	
	def __call__():	
	"""" Calculates the corrections to the dipole moment
    
    uncorrected_property: The uncorrected dipole moment
    n_nm: The number of normal modes of the molecule
    pre_property: The second derivative of the dipole moment
    return: The corrections to the dipole moment, the corrected dipole 
    moment as np.arrays 
    """
		correction_property = zeros((3))
		eig = absolute(eig)
		
		for i in range(n_nm):
			factor = 1/(sqrt(eig[i])) # the reduced one
			correction_property[0] += pre_property[i, 0]*factor
			correction_property[1] += pre_property[i, 1]*factor
			correction_property[2] += pre_property[i, 2]*factor
		
		correction_property = correction_property * prefactor
		corrected_property = add(uncorrected_property, correction_property)
		
		return correction_property, corrected_property  

class Property_2_Tensor(Property):	
	def __call__():
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
	def __call__():
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
