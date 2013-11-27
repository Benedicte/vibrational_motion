
from numpy import array, dot, sqrt, set_printoptions, reshape, multiply, divide, add, subtract, diag
import numpy as np
from scipy import mat, linalg, double
from read_input import *
from abavib import *

dipole_h20 = mat([[0.083445,0.020685,-0.008117] 
,[-0.123263,-0.080377,0.031540]
,[-0.366672,-0.483183,0.189596]])

correct_EVEC = mat([[-0.00131353,-0.00001741,0.00029587,-0.00016271,0.00000038,0.00006501]
,[0.00007785,-0.00060863,-0.00084065,-0.00064259,-0.00032658,0.00406074]
,[-0.00018153,0.00151054,0.00052282,-0.000208,-0.00088988,-0.00002127]
,[0.00116367,0.00047638,-0.00027149,-0.00042556,0.00006439,-0.00006238]
,[0.00000617,-0.0007995,0.00059862,-0.0006688,0.00040987,-0.00405949]
,[0.00036161,-0.00151616,0.00016393,-0.00002019,-0.00098291,-0.00002758]
,[0.01633121,-0.00137943,0.00208677,0.00294594,0.00196957,0.00042724]
,[-0.00289218,0.00884232,0.01679361,0.01035385,0.00450935,0.00316981]
,[0.00144209,-0.00874834,-0.00591666,0.01308062,0.01362057,0.00033726]
,[-0.01395275,-0.00590483,-0.00247371,0.00639033,-0.0029974,-0.00046896]
,[0.00155876,0.01350575,-0.01295232,0.01045877,-0.0058313,-0.00318957]
,[-0.00430002,0.00883742,-0.0049825,-0.00945915,0.01610197,0.00043797]])

correct_cff =  mat([[-4.9325820e-06,-4.5992660e-07,1.7554510e-06,-1.2019920e-06,-1.3271570e-05,-9.4579700e-07]
,[-4.5992660e-07,-9.2274080e-06,-4.8430190e-06,6.9665310e-06,4.3085360e-06,2.3705340e-07]
,[1.7554510e-06,-4.8430190e-06,-6.1919390e-06,2.1113940e-06,2.8311200e-06,2.9884780e-07]
,[-1.2019920e-06,6.9665310e-06,2.1113940e-06,-1.1614660e-05,-9.3882990e-06,-2.4648970e-07]
,[-1.3271570e-05,4.3085360e-06,2.8311200e-06,-9.3882990e-06,-1.3155740e-05,-6.8674150e-07]
,[-9.4579700e-07,2.3705340e-07,2.9884780e-07,-2.4648970e-07,-6.8674150e-07,1.2495880e-07]])


correct_big_EVEC = mat([[-0.00131353,-0.00001741,0.00029587,-0.00016271,0.00000038,0.00006501,0.00027175,0.00568992,-0.00000074,-0.00001015,0.00000448,-0.00006349]
,[0.00007785,-0.00060863,-0.00084065,-0.00064259,-0.00032658,0.00406074,-0.00000384,-0.00000448,0.00401866,-0.00016465,0.00016675,0.00001509]
,[-0.00018153,0.00151054,0.00052282,-0.000208,-0.00088988,-0.00002127,-0.00004001,-0.00007291,0.00003336,0.0001495,0.0055534,-0.00005365]
,[0.00116367,0.00047638,-0.00027149,-0.00042556,0.00006439,-0.00006238,0.00569633,0.00000134,0.00000126,0.00004064,-0.00003141,0.00004909]
,[0.00000617,-0.0007995,0.00059862,-0.0006688,0.00040987,-0.00405949,-0.00000384,-0.00000448,0.00401866,-0.00016465,0.00016675,0.00001509]
,[0.00036161,-0.00151616,0.00016393,-0.00002019,-0.00098291,-0.00002758,0.00003536,0.00007269,-0.00003299,0.00553274,0.00009811,-0.00049504]
,[0.01633121,-0.00137943,0.00208677,0.00294594,0.00196957,0.00042724,-0.00304474,0.00405091,0.00000712,0.00015899,0.00131398,0.01520468]
,[-0.00289218,0.00884232,0.01679361,0.01035385,0.00450935,0.00316981,0.00137169,-0.0013139,0.00397985,0.00303638,-0.00307331,-0.00021873]
,[0.00144209,-0.00874834,-0.00591666,0.01308062,0.01362057,0.00033726,0.0010712,0.00085591,0.00003751,-0.00053149,0.00562937,-0.0063333]
,[-0.01395275,-0.00590483,-0.00247371,0.00639033,-0.0029974,-0.00046896,0.0036063,-0.00292382,0.00000957,0.00022126,0.00126997,0.0153427]
,[0.00155876,0.01350575,-0.01295232,0.01045877,-0.0058313,-0.00318957,-0.00129009,0.00147743,0.00397887,0.00301146,-0.0030557,-0.00027397]
,[-0.00430002,0.00883742,-0.0049825,-0.00945915,0.01610197,0.00043797,-0.00107585,-0.00085612,-0.00003714,0.00621373,0.00002214,0.00578462]])

dipole = array([[-0.048779,-0.282926,-0.008477],
[0.048120,0.280315,-0.020823],
[0.000943,0.000917,-0.080650],
[-0.001330,-0.001029,-0.137956],
[0.000026,0.000180,-0.002133],
[-0.000336,0.000407,-0.240555]])

EVAL = array([0.0003967267, 0.0003909715, 5.5175184e-005, 4.4395569e-005, 2.8355625e-005, 1])
EVAL1 = array([0.0003967267, 0.0003909715, 5.5175184e-005, 4.4395569e-005, 2.8355625e-005, 0.00028389])

DIPOLE_PRE = array([0.00026608, -0.00020134, 0.97738028]) 

dipole_pre = mat([ 0.37370174, 0.49133014, -0.19279329])

shield_correct = array([[[-2.73543218 , 1.29310207 , -1.89858102]
,[ 0.23201889 , -11.10707143 , -0.74269857]
,[-1.35752979 , 0.82513205 , -4.01976529]]
,[[ -2.75685668 , 1.31036411 , 1.85157732]
,[ 0.21430005 , -11.22126964 , 0.71749018]
,[ 1.31818971 , -0.84225704 , -3.98227118]]
,[[ -0.2278518 , 0.00750604 , -0.12086826]
,[ -0.00043512 , -0.04561196 , -0.11442797]
,[ -0.09115688 , 0.03692163 , -0.34873231]]
,[[ -0.2255517 , 0.00613924 , 0.1193651]
,[ -0.00031229 , -0.04107803 , 0.1134338]
,[ 0.08987755 , -0.03594509 , -0.34278067]]])
   
correct_MOLQUAD = array([-0.00825547,-0.00810764,0.0000083,0.02245658,-0.00003619,-0.01420111])
correct_g_tensor = array([[-0.02776227,-0.00001852,0.00232808],[-0.00000333,-0.00035603,-0.00000795],[0.00046495,-0.00000757,0.00136186]])
correct_magnet = array([-0.03198422,-0.00377751,0.00008762,0.0440208,0.00003563,0.00310904])
correct_polari = array([0.02969292,0.03673433,-0.00014646,0.18001732,-0.0001547,0.08188467])
correct_spinrot = array([[-0.21388431,0.0334947,0.00389405,-0.09999148,0.04560931,0.00144877,-0.05558241,0.00337925,-0.03279227]
,[-0.1955979,-0.03304894,0.0032995,0.10214155,0.0453841,-0.00085245,-0.05945283,-0.0027541,-0.03244115]])

correct_nucquad = array([[0.00173516,-0.00266833,-0.00676687,0.01132919,-0.01183273,-0.01306435]
,[0.00177681,-0.0026176,0.00675537,0.01120484,0.01177362,-0.01298165]
,[-0.00844869,0.00204556,0.01162262,-0.00631312,0.00461045,0.01476181]
,[-0.00845538,0.00208702,-0.01167536,-0.00643277,-0.00467444,0.01488815]])

def main():
    
    input_name = "input_h2o2/"
    mol_name = input_name + 'MOLECULE.INP'
    hessian_name = input_name + 'hessian'
    cff_name = input_name + 'cubic_force_field'
    coordinates, masses,  num_atoms_list, charge_list, n_atoms = read_molecule(mol_name)

    n_coords = 3 * n_atoms
    n_nm = n_coords - 6 
    
    hessian = read_hessian(hessian_name, n_coords)
    hessian_t = hessian.transpose()
    hessian_temp = add(hessian, hessian_t) 
    hessian = subtract(hessian_temp , diag(hessian.diagonal()))
    
    eig, eigvec, freq, eigvec_full = fundamental_freq(hessian, num_atoms_list, charge_list, coordinates, n_atoms)
    cubic_force_field = read_cubic_force_field(cff_name, n_coords) 
    cff_norm, cff_norm_reduced = to_normal_coordinates_3D(cubic_force_field, correct_big_EVEC, n_atoms)
    cff_norm1, cff_norm_reduced1 = to_normal_coordinates_3D(cubic_force_field, eigvec_full, n_atoms)

    effective_geometry_norm = effective_geometry(cff_norm_reduced, freq, n_atoms)
    effective_geometry_cart = to_cartessian_coordinates(effective_geometry_norm, n_atoms, eigvec)
    
    dipole_moment_diff, dipole_moment_corrected = get_dipole_moment(dipole, n_nm, eig, dipole_pre, True)
   
    shield_deriv, prop_type = read_4d_input(input_name + "SHIELD", 4, 6)
    shield = get_4D_property("Shield", shield_deriv, n_nm, n_atoms, EVAL, True)
    
    nuc_quad_deriv, prop_type = read_nucquad(input_name + "NUCQUAD", 4, 6)
    nuc_quad = get_4D_property(prop_type, nuc_quad_deriv, n_nm, n_atoms, EVAL, True)
    
    #spin_spin_deriv, prop_type = read_3d_input(input_name + "SPIN-SPIN", 6)
    #spin_spin = get_3D_property(prop_type, spin_spin_deriv, n_nm, EVAL, True)
    #mol_quad = get_3D_property(prop_type, mol_quad_deriv, n_nm, eig, True)
    
    mol_quad_deriv, prop_type = read_mol_quad(input_name + "MOLQUAD", 6)
    mol_quad = get_3D_property(prop_type, mol_quad_deriv, 6, EVAL, True)
    
    magnet_deriv, g_tensor_deriv = read_magnet(input_name + "MAGNET", 6)
    g_tensor = get_3D_property("g-tensor", g_tensor_deriv, n_nm, EVAL, True)  
    magnet = get_3D_property("magnet", magnet_deriv, n_nm, EVAL, True)

    spinrot_deriv, prop_type = read_spinrot(input_name + "SPIN-ROT", 4, 6)
    spinrot = get_4D_property(prop_type, spinrot_deriv, n_nm, n_atoms, EVAL, True) 
    
    quartic_force_field = read_quartic_force_field(input_name + 'quartic',12) 
    
    print "g-tensor"
    print g_tensor   
    print correct_g_tensor
    

set_printoptions(suppress=True) #Avoid scientific notation when printing arrs

if __name__ == '__main__':
    main()


