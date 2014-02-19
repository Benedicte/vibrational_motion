import Molecule as mol
import numpy as np
import pydoc

class Property:
    """The superclass for calculating properties of a molecule"""
    
    def __init__(self, molecule, property_name):
        self.m_e = 1822.8884796 # conversion factor from a.m.u to a.u 
        self.prefactor = 1/(4*self.m_e)
        self.molecule = molecule
                
    def __call__():
        raise NotImplementedError\
        ('___call__ missing in class %s' % self.__class__.__name__)
    
    def quartic_precision():
        """"If the quartic force field precision is derired, this function can be used 
        inside the call function where the new terms is simply added as a correction"""
        
        

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
        corrected_property = np.around(self.corrected_property, decimals=4)
        correction_property = np.around(self.correction_property, decimals=4)
        uncorrected_property = np.around(self.uncorrected_property, decimals=4)
        atom_list = self.molecule.atom_list 
        
        print(corrected_property)
        #f.write("& & Effective geometry &  $<P^{(0)}_2>_{eff}$ &  Vibrationally corrected \\\\" + "\n") 
        if(corrected_property.ndim == 1):
            #line = str(results).strip('[]')
            #line = re.sub("\\s+", "&", line)
            #line = line + "\\\\"
            f.write(property_type) 
            f.write("& X & " + str(uncorrected_property[0]) + "&" + str(correction_property[0]) + "&" +str(corrected_property[0])+"\\\\  \n")
            f.write("& Y &" + str(uncorrected_property[1]) + "&" +str(correction_property[1]) + "&" +str(corrected_property[1])+"\\\\  \n")
            f.write("& Z &" + str(uncorrected_property[2]) + "&" +str(correction_property[2]) + "&" +str(corrected_property[2])+"\\\\  \n")
            #f.write(line + "\n")
            f.write("\hline" + "\n")
            f.close()
        
        if(corrected_property.ndim == 2):
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
        
