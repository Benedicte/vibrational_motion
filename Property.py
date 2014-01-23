class Property:
"""The superclass for calculating properties of a molecule"""

def __init__(property_type, nm, eig)
	m_e = 1822.8884796 # conversion factor from a.m.u to a.u 
    prefactor = 1/(4*m_e)
    
def __call__()
	raise NotImplementedError\
	('___call__ missing in class %s' % self.__class__.__name__)

def write_to_file(molecule, property_type, results, n_atom = None):
    
        filename = "output/" + molecule
        f = open(filename, "a")
        f.write(property_type + "\n")
        
        if(results.ndim == 1):
            line = str(results).strip('[]')
            f.write(line + "\n")
            f.close()
        
        if(results.ndim == 2):
            line1 = str(results[0]).strip('[]')
            line2 = str(results[1]).strip('[]')
            line3 = str(results[2]).strip('[]')
            
            f.write(line1 + "\n")
            f.write(line2 + "\n")
            f.write(line3 + "\n")

            f.close()

        if(results.ndim == 3):        
            for atom in range(n_atom):
                line1 = str(results[atom][0]).strip('[]')
                line2 = str(results[atom][1]).strip('[]')
                line3 = str(results[atom][2]).strip('[]')
            
                f.write(line1 + "\n")
                f.write(line2 + "\n")
                f.write(line3 + "\n")
                
                f.write("\n") # Seperates the 2D matrices making up the 3D matrix
