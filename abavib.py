import read_input as ri
import Molecule as mol
import Propertyclasses as pr
import pydoc

"""
The module working as the interface of the command line based program. 
Initiated the other modules and funcions needed to perform the 
calculations requested.  
"""

if __name__ == '__main__':

    print("In order to calculate molecular properties, make sure to have made a directory named\
    input_(molecule_name) where a copy of the MOLECULE.INP, hessian, and cubic_force_field\
    files from DALTON are present. In addition to this a copy of the current propety run\
    in DALTON is needed for every property which is to be calculated.")

    print("")
    print("")

    molecule_name = raw_input('Which molecule should calculations be made for? (ex. h2o)')
    if (molecule_name == ""):
        molecule_name = "fluoromethane"
        molecule_name = "fluoromethane"
    
    dft = raw_input('at DFT or at HF level?')
    
    #if(dft == "dft"):
    molecule_name = "dft_" + molecule_name
    
    molecule = mol.Molecule(molecule_name)


    prop_name = raw_input('Which property should be calculated? For options enter "opt"')
    
    if(prop_name == "opt"):
        print("Dipole Moment")
        print("Magnetizability")
        print("g-factor")
        print("Nuclear spin-rotation")
        print("Molecular quadropole moment")
        print("Spin-spin coupling")
        print("Polarizability")
        print("Nuclear shielding")
        print("Nuclear spin correction")
        print("Nuclear quadropole moment")
        print("Optical rotation")
    
    elif(prop_name == "Dipole Moment"):
        prop = pr.Property_1_Tensor(molecule, prop_name)
    elif(prop_name == "Magnetizability"):
        prop = pr.Property_2_Tensor(molecule, prop_name)
    elif(prop_name == "g-factor"):
        prop = pr.Property_2_Tensor(molecule, prop_name)
    elif(prop_name == "Nuclear spin-rotation"):
        prop = pr.Property_3_Tensor(molecule, prop_name)
    elif(prop_name == "Molecular quadropole moment"):
        prop = pr.Property_2_Tensor(molecule, prop_name)
    elif(prop_name == "Spin-spin coupling"):
        prop = pr.Property_2_Tensor(molecule, prop_name)
    elif(prop_name == "Polarizability"):
        prop = pr.Polarizability(molecule, prop_name)
    elif(prop_name == "Nuclear shielding"):
        prop = pr.Property_3_Tensor(molecule, prop_name)
    elif(prop_name == "Nuclear spin correction"):
        prop = pr.Property_3_Tensor(molecule, prop_name)
    elif(prop_name == "Nuclear quadropole moment"):
        prop = pr.Property_3_Tensor(molecule, prop_name)
    elif(prop_name == "Optical rotation"):
        prop = pr.Property_2_Tensor(molecule, prop_name)
    elif(prop_name == "all"):
        prop = pr.Property_1_Tensor(molecule, "Dipole Moment")
        prop()
        prop = pr.Property_3_Tensor(molecule, "Nuclear spin-rotation")
        prop()
        prop = pr.Property_2_Tensor(molecule, "Molecular quadropole moment")
        prop()
        prop = pr.Property_3_Tensor(molecule, "Nuclear shielding")
        prop()
        prop = pr.Property_3_Tensor(molecule, "Nuclear quadropole moment")

    else:
        print ("Not a supported property")
        
    prop()

