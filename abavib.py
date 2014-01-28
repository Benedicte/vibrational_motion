import read_input as ri
import Molecule as mol
import Property as pr
import pydoc

if __name__ == '__main__':

    print("In order to calculate molecular properties, make sure to have made a directory named\
    input_(molecule_name) where a copy of the MOLECULE.INP, hessian, and cubic_force_field\
    files from DALTON are present. In addition to this a copy of the current propety run\
    in DALTON is needed for every property which is to be calculated.")

    print("")
    print("")

    molecule_name = raw_input('Which molecule should calculations be made for? (ex. h2o)')
    molecule = mol.Molecule(molecule_name)

    prop = raw_input('Which property should be calculated?')

    if(prop == "Dipole Moment"):
        prop = pr.Property_1_Tensor(molecule)
    elif(prop == "Magnetizability"):
        prop = pr.Property_1_Tensor(molecule)
    elif(prop == "g-factor"):
        prop = pr.Property_1_Tensor(molecule)
    elif(prop == "Nuclear spin-rotation"):
        prop = pr.Property_1_Tensor(molecule)
    elif(prop == "Molecular quadropole moment"):
        prop = pr.Property_1_Tensor(molecule)
    elif(prop == "Spin-spin coupling"):
        prop = pr.Property_1_Tensor(molecule)
    elif(prop == "Polarizability"):
        prop = pr.Property_1_Tensor(molecule)
    elif(prop == "Nuclear shielding"):
        prop = pr.Property_1_Tensor(molecule)
    elif(prop == "Nuclear spin correction"):
        prop = pr.Property_1_Tensor(molecule)
    elif(prop == "Nuclear quadropole moment"):
        prop = pr.Property_1_Tensor(molecule)
    elif(prop == "Optical rotation"):
        prop = pr.Property_1_Tensor(molecule)
    else:
        print ("Not a supported property")



    #input_name = "input_" + molecule + "/"
    #output_file_name = "output/" + molecule
    #open(output_file_name, 'w').close() # As we are appending to the output, the old results must be deleted before each run

    #unittest.main()
