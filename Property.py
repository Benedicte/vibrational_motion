"""
Copyright (c) 2013-2014 Benedicte Ofstad
Distributed under the GNU Lesser General Public License v3.0. 
For full terms see the file LICENSE.md.
"""

import Molecule as mol
import numpy as np
import read_input as ri
import pydoc

class Property:
    """The superclass for calculating properties of a molecule"""
    
    def __init__(self, molecule, property_name):
        self.m_e = 1822.8884796 # conversion factor from a.m.u to a.u 
        self.prefactor = 1/(4*self.m_e)
        self.molecule = molecule
        self.freq = self.molecule.frequencies
                
    def __call__():
        """The superclass for the call function, raises a NotImplementedError."""
        raise NotImplementedError\
        ('___call__ missing in class %s' % self.__class__.__name__)
    
    def quartic_precision(cff_norm, qff_norm, prop_deriv):
        """The superclass for the quartic_precision function, raises a NotImplementedError."""
        raise NotImplementedError\
        ('___call__ missing in class %s' % self.__class__.__name__)

    def get_quartic_force_field(self):
        """Returns the quartic force field in cartessian coordinates as 
        a 4D np.array"""
        qff = ri.read_quartic_force_field1(self.molecule.input_name + "/quartic", self.molecule.n_coordinates)
        return(qff)
        
    def write_to_file(self, property_type, n_atom = None):
        """ Writes the resutls to file. It writes the uncorrected property
        the property corrections and the corrected property. Instead of 
        producing its own lable. This method writes the table in LaTeX code. 
        
        property_type: This will be written in the header of the table.
        n_atom: The number of atoms making up the molecule.
        
        return: A file with the results written out in LaTeX code. 
        
        """
        
        filename = self.molecule.get_output_name()
        f = open(filename, "a")
        atom_list = self.molecule.atom_list 
        
        #f.write("& & Effective geometry &  $<P^{(0)}_2>_{eff}$ &  Vibrationally corrected \\\\" + "\n") 
        
        corrected_property = np.around(self.corrected_property, decimals=4)
        correction_property = np.around(self.correction_property, decimals=4)
        uncorrected_property = np.around(self.uncorrected_property, decimals=4)
        f.write(property_type) 
        
        if(property_type == "Optical rotation"):
            
            corrected_property = np.around(self.corrected_property1, decimals=4)
            correction_property = np.around(self.correction_property1, decimals=4)
            uncorrected_property = np.around(self.uncorrected_property1, decimals=4)
            f.write(property_type)
        
            f.write( "& XX " + "&"+ str(uncorrected_property[0][0]) + "&" + str(correction_property[0][0]) + "&" +str(corrected_property[0][0])+ "\\\\ \n")
            f.write( "& XY " + "&"+ str(uncorrected_property[0][1]) + "&" + str(correction_property[0][1]) + "&" +str(corrected_property[0][1])+ "\\\\ \n")
            f.write( "& XZ " + "&"+ str(uncorrected_property[0][2]) + "&" + str(correction_property[0][2]) + "&" +str(corrected_property[0][2])+ " \\\\ \n")
            f.write(" & YX " + "&"+ str(uncorrected_property[1][0]) + "&" + str(correction_property[1][0]) + "&" +str(corrected_property[1][0])+ "\\\\ \n")
            f.write(" & YY " + "&"+ str(uncorrected_property[1][1]) + "&" + str(correction_property[1][1]) + "&" +str(corrected_property[1][1])+ "\\\\ \n")
            f.write(" & YZ " + "&"+ str(uncorrected_property[1][2]) + "&" + str(correction_property[1][2]) + "&" +str(corrected_property[1][2])+ "\\\\ \n")
            f.write(" & ZX " + "&" +str(uncorrected_property[2][0]) + "&" + str(correction_property[2][0]) + "&" +str(corrected_property[2][0])+ "\\\\ \n")
            f.write(" & ZY  " + "&"+ str(uncorrected_property[2][1]) + "&" + str(correction_property[2][1]) + "&" +str(corrected_property[2][1])+ "\\\\ \n")
            f.write(" & ZZ "+ "&"+ str(uncorrected_property[2][2]) + "&" + str(correction_property[2][2]) + "&" +str(corrected_property[2][2]) + "\\\\ \n")
            
            
            f.write("\hline" + "\n")
            
            corrected_property = np.around(self.corrected_property2, decimals=4)
            correction_property = np.around(self.correction_property2, decimals=4)
            uncorrected_property = np.around(self.uncorrected_property2, decimals=4)
            f.write(property_type)
        
            f.write( "& XX " + "&"+ str(uncorrected_property[0][0]) + "&" + str(correction_property[0][0]) + "&" +str(corrected_property[0][0])+ "\\\\ \n")
            f.write( "& XY " + "&"+ str(uncorrected_property[0][1]) + "&" + str(correction_property[0][1]) + "&" +str(corrected_property[0][1])+ "\\\\ \n")
            f.write( "& XZ " + "&"+ str(uncorrected_property[0][2]) + "&" + str(correction_property[0][2]) + "&" +str(corrected_property[0][2])+ " \\\\ \n")
            f.write(" & YX " + "&"+ str(uncorrected_property[1][0]) + "&" + str(correction_property[1][0]) + "&" +str(corrected_property[1][0])+ "\\\\ \n")
            f.write(" & YY " + "&"+ str(uncorrected_property[1][1]) + "&" + str(correction_property[1][1]) + "&" +str(corrected_property[1][1])+ "\\\\ \n")
            f.write(" & YZ " + "&"+ str(uncorrected_property[1][2]) + "&" + str(correction_property[1][2]) + "&" +str(corrected_property[1][2])+ "\\\\ \n")
            f.write(" & ZX " + "&" +str(uncorrected_property[2][0]) + "&" + str(correction_property[2][0]) + "&" +str(corrected_property[2][0])+ "\\\\ \n")
            f.write(" & ZY  " + "&"+ str(uncorrected_property[2][1]) + "&" + str(correction_property[2][1]) + "&" +str(corrected_property[2][1])+ "\\\\ \n")
            f.write(" & ZZ "+ "&"+ str(uncorrected_property[2][2]) + "&" + str(correction_property[2][2]) + "&" +str(corrected_property[2][2]) + "\\\\ \n")
            
            corrected_property = np.around(self.corrected_property3, decimals=4)
            correction_property = np.around(self.correction_property3, decimals=4)
            uncorrected_property = np.around(self.uncorrected_property3, decimals=4)
            f.write(property_type)
        
            f.write( "& XX " + "&"+ str(uncorrected_property[0][0]) + "&" + str(correction_property[0][0]) + "&" +str(corrected_property[0][0])+ "\\\\ \n")
            f.write( "& XY " + "&"+ str(uncorrected_property[0][1]) + "&" + str(correction_property[0][1]) + "&" +str(corrected_property[0][1])+ "\\\\ \n")
            f.write( "& XZ " + "&"+ str(uncorrected_property[0][2]) + "&" + str(correction_property[0][2]) + "&" +str(corrected_property[0][2])+ " \\\\ \n")
            f.write(" & YX " + "&"+ str(uncorrected_property[1][0]) + "&" + str(correction_property[1][0]) + "&" +str(corrected_property[1][0])+ "\\\\ \n")
            f.write(" & YY " + "&"+ str(uncorrected_property[1][1]) + "&" + str(correction_property[1][1]) + "&" +str(corrected_property[1][1])+ "\\\\ \n")
            f.write(" & YZ " + "&"+ str(uncorrected_property[1][2]) + "&" + str(correction_property[1][2]) + "&" +str(corrected_property[1][2])+ "\\\\ \n")
            f.write(" & ZX " + "&" +str(uncorrected_property[2][0]) + "&" + str(correction_property[2][0]) + "&" +str(corrected_property[2][0])+ "\\\\ \n")
            f.write(" & ZY  " + "&"+ str(uncorrected_property[2][1]) + "&" + str(correction_property[2][1]) + "&" +str(corrected_property[2][1])+ "\\\\ \n")
            f.write(" & ZZ "+ "&"+ str(uncorrected_property[2][2]) + "&" + str(correction_property[2][2]) + "&" +str(corrected_property[2][2]) + "\\\\ \n")

            f.close()
            
            return
        
        if(corrected_property.ndim == 1):
            #line = str(results).strip('[]')
            #line = re.sub("\\s+", "&", line)
            #line = line + "\\\\"
            
            f.write("& X & " + str(uncorrected_property[0]) + "&" + str(correction_property[0]) + "&" +str(corrected_property[0])+"\\\\  \n")
            f.write("& Y &" + str(uncorrected_property[1]) + "&" +str(correction_property[1]) + "&" +str(corrected_property[1])+"\\\\  \n")
            f.write("& Z &" + str(uncorrected_property[2]) + "&" +str(correction_property[2]) + "&" +str(corrected_property[2])+"\\\\  \n")
            #f.write(line + "\n")
            f.write("\hline" + "\n")
            f.close()
        
        if(corrected_property.ndim == 2):
        
            f.write( "& XX " + "&"+ str(uncorrected_property[0][0]) + "&" + str(correction_property[0][0]) + "&" +str(corrected_property[0][0])+ "\\\\ \n")
            f.write( "& XY " + "&"+ str(uncorrected_property[0][1]) + "&" + str(correction_property[0][1]) + "&" +str(corrected_property[0][1])+ "\\\\ \n")
            f.write( "& XZ " + "&"+ str(uncorrected_property[0][2]) + "&" + str(correction_property[0][2]) + "&" +str(corrected_property[0][2])+ " \\\\ \n")
            f.write(" & YX " + "&"+ str(uncorrected_property[1][0]) + "&" + str(correction_property[1][0]) + "&" +str(corrected_property[1][0])+ "\\\\ \n")
            f.write(" & YY " + "&"+ str(uncorrected_property[1][1]) + "&" + str(correction_property[1][1]) + "&" +str(corrected_property[1][1])+ "\\\\ \n")
            f.write(" & YZ " + "&"+ str(uncorrected_property[1][2]) + "&" + str(correction_property[1][2]) + "&" +str(corrected_property[1][2])+ "\\\\ \n")
            f.write(" & ZX " + "&" +str(uncorrected_property[2][0]) + "&" + str(correction_property[2][0]) + "&" +str(corrected_property[2][0])+ "\\\\ \n")
            f.write(" & ZY  " + "&"+ str(uncorrected_property[2][1]) + "&" + str(correction_property[2][1]) + "&" +str(corrected_property[2][1])+ "\\\\ \n")
            f.write(" & ZZ "+ "&"+ str(uncorrected_property[2][2]) + "&" + str(correction_property[2][2]) + "&" +str(corrected_property[2][2]) + "\\\\ \n")
            
            
            f.write("\hline" + "\n")

            f.close()

        if(corrected_property.ndim == 3): 
            
            f.write(property_type + "\\\\" + "\n")       
            for atom in range(n_atom):
                f.write("\hline"+ "\n" )
                f.write("Atom: "+ atom_list[atom] ) 
                
                f.write( "& XX " + "&"+ str(uncorrected_property[atom][0][0]) + "&" + str(correction_property[atom][0][0]) + "&" +str(corrected_property[atom][0][0])+ "\\\\ \n")
                f.write( "& XY " + "&"+ str(uncorrected_property[atom][0][1]) + "&" + str(correction_property[atom][0][1]) + "&" +str(corrected_property[atom][0][1])+ "\\\\ \n")
                f.write( "& XZ " + "&"+ str(uncorrected_property[atom][0][2]) + "&" + str(correction_property[atom][0][2]) + "&" +str(corrected_property[atom][0][2])+ " \\\\ \n")
                f.write(" & YX " + "&"+ str(uncorrected_property[atom][1][0]) + "&" + str(correction_property[atom][1][0]) + "&" +str(corrected_property[atom][1][0])+ "\\\\ \n")
                f.write(" & YY " + "&"+ str(uncorrected_property[atom][1][1]) + "&" + str(correction_property[atom][1][1]) + "&" +str(corrected_property[atom][1][1])+ "\\\\ \n")
                f.write(" & YZ " + "&"+ str(uncorrected_property[atom][1][2]) + "&" + str(correction_property[atom][1][2]) + "&" +str(corrected_property[atom][1][2])+ "\\\\ \n")
                f.write(" & ZX " + "&"+ str(uncorrected_property[atom][2][0]) + "&" + str(correction_property[atom][2][0]) + "&" +str(corrected_property[atom][2][0])+ "\\\\ \n")
                f.write(" & ZY  " + "&"+ str(uncorrected_property[atom][2][1]) + "&" + str(correction_property[atom][2][1]) + "&" +str(corrected_property[atom][2][1])+ "\\\\ \n")
                f.write(" & ZZ "+ "&"+ str(uncorrected_property[atom][2][2]) + "&" + str(correction_property[atom][2][2]) + "&" +str(corrected_property[atom][2][2]) + "\\\\ \n")
                
                f.write("\n") # Seperates the 2D matrices making up the 3D matrix

            f.write("\hline" + "\n")
            #f.write("\\end{tabular}")
            f.close()
        
