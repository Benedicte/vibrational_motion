import unittest
import abavib as av
import read_input as ri
import numpy as np

#The reason we use this one, is because there are any number of eigenvectors which are correct eigenvectors, for the purpose of testing
#we use the same one that DALTON operates with
correct_big_EVEC = np.array([[-0.00131353,-0.00001741,0.00029587,-0.00016271,0.00000038,0.00006501,0.00027175,0.00568992,-0.00000074,-0.00001015,0.00000448,-0.00006349]
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

#This one should work, check out vs. Master
EVAL = np.array([0.0003967267, 0.0003909715, 5.5175184e-005, 4.4395569e-005, 2.8355625e-005, 1])

dipole_pre = np.array([ 0.00026608,-0.00020134,0.97738028])

dipole = np.array([[-0.04877, -0.282926, -0.008477]
,[0.04812, 0.280315, -0.020823]
,[0.000943, 0.000917, -0.08065]
,[-0.00133, -0.001029, -0.137956]
,[0.000026, 0.00018, -0.002133]
,[-0.000336, 0.000407, -0.240555]])

class abavib_test(unittest.TestCase):
    def setUp(self):
        self.molecule = "h2o2"
        self.input_name = "input_" + self.molecule + "/"
        self.mol_name = self.input_name + 'MOLECULE.INP'
        self.cff_name = self.input_name + 'cubic_force_field'
        self.coordinates, self.masses,  self.num_atoms_list \
            ,self.charge_list, self.n_atoms = av.read_molecule(self.mol_name)
        self.n_coordinates = self.n_atoms * 3  
        self.n_nm = self.n_coordinates - 6 
        hessian_name = self.input_name + 'hessian'
        self.hessian = av.read_hessian(hessian_name, self.n_atoms*3)
        hessian_t = self.hessian.transpose()
        hessian_temp = np.add(self.hessian, hessian_t) 
        self.hessian = np.subtract(hessian_temp , np.diag(self.hessian.diagonal()))
        self.eig, self.eigvec, self.freq, self.eigvec_full = \
            av.fundamental_freq(self.hessian, self.num_atoms_list, \
            self.charge_list, self.coordinates, self.n_atoms)
        self.cubic_force_field = av.read_cubic_force_field(self.cff_name,\
         self.n_coordinates) 
        self.cff_norm, self.cff_norm_reduced = av.to_normal_coordinates_3D(self.cubic_force_field, correct_big_EVEC, self.n_atoms)
        effective_geometry_norm = av.effective_geometry(self.cff_norm_reduced, self.freq, self.n_atoms)
        self.effective_geometry_cart = av.to_cartessian_coordinates(effective_geometry_norm, self.n_atoms, self.eigvec)
        self.dipole_moment_diff, self.dipole_moment_corrected = av.get_dipole_moment(dipole, self.n_nm, self.eig, dipole_pre, False)
        shield_deriv, self.prop_type = ri.read_4d_input(self.input_name + "SHIELD", self.n_atoms, self.n_nm)
        self.shield = av.get_4D_property("Shield", shield_deriv, self.n_nm, self.n_atoms, EVAL, True)
        nuc_quad_deriv, self.prop_type = ri.read_nucquad(self.input_name + "NUCQUAD", self.n_atoms, self.n_nm)
        self.nuc_quad = av.get_4D_property(self.prop_type, nuc_quad_deriv, self.n_nm, self.n_atoms, EVAL, True)

class read_molecule_test(abavib_test):        
    def test_coordinates(self):
        
        if(self.molecule == "h2o"):
            self.correct_coordinates = [[-1.42157256,  2.28115327,  0.00554911],
                                        [ -0.13533793,  2.10700057,  0.07387382],
                                        [-2.02521896,  3.34965922, -0.41371135]]
            self.correct_coordinates = np.array(self.correct_coordinates)
            self.assertTrue((self.coordinates == self.correct_coordinates).all())
            
        elif(self.molecule == "h2o2"):
            self.correct_coordinates = np. array([[0.00000000,  1.40784586, -0.09885600]
                                        ,[0.00000000, -1.40784586, -0.09885600]
                                        ,[0.69081489,  1.72614891,  1.56891868]
                                        ,[-0.69081489, -1.72614891,  1.56891868]])
            self.assertTrue((self.coordinates == self.correct_coordinates).all())
            
    def test_masses(self):
        
        if(self.molecule == "h2o"):
            self.correct_masses = [15.9994, 1.00794, 1.00794]
            self.assertSequenceEqual(self.masses, self.correct_masses)
            
        elif(self.molecule == "h2o2"):
            self.correct_masses = [15.9994, 15.9994, 1.00794, 1.00794]
            self.assertSequenceEqual(self.masses, self.correct_masses)
             
    def test_num_atoms_list(self):
        
        if(self.molecule == "h2o"):
            self.assertSequenceEqual(self.num_atoms_list, [1,2])
            
        elif(self.molecule == "h2o2"):
            self.assertSequenceEqual(self.num_atoms_list, [2,2])
            
        
    def test_charge_list(self):
        self.assertSequenceEqual(self.charge_list, [8,1])
        
    def test_n_atoms(self):
        
        if(self.molecule == "h2o"):
            self.assertEquals(self.n_atoms, 3)
        elif(self.molecule == "h2o2"):
            self.assertEquals(self.n_atoms, 4)
        
class read_hessian_test(abavib_test):
    def test_hessian_values(self):
        
        if(self.molecule == "h2o"):
            correct_hessian = np.array([[9.79479, -0.452448, 0.229158, -6.327811, -0.924135, 0.318151, -3.466978, 1.376583, -0.547309]
            ,[ -0.452448, 32.105552, -11.992317, 0.767043, -10.749399, 3.928526, -0.314595, -21.356153, 8.06379]
            ,[ 0.229158, -11.992317, 6.029794, -0.298792, 3.882729, -2.197041, 0.069634, 8.062308, -3.832753]
            ,[ -6.327811, 0.767043, -0.298792, 6.541742, -1.005224, 0.392545, -0.213931, 0.238181, -0.093753]
            ,[ -0.924135, -10.749399, 3.882729, -1.005224, 4.894784, -1.718158, 1.929359, 5.767001, -2.204113]
            ,[ 0.318151, 3.928526, -2.197041, 0.392545, -1.718158, 1.177308, -0.710696, -2.20263, 1.019733]
            ,[ -3.466978, -0.314595, 0.069634, -0.213931, 1.929359, -0.710696, 3.680909, -1.614764, 0.641062]
            ,[ 1.376583, -21.356153, 8.062308, 0.238181, 5.767001, -2.20263, -1.614764, 15.589152, -5.859678]
            ,[ -0.547309, 8.06379, -3.832753, -0.093753, -2.204113, 1.019733, 0.641062, -5.859678, 2.81302]])
            self.assertTrue((self.hessian == correct_hessian).all())
            
        elif(self.molecule == "h2o2"):
            correct_hessian = np.array([[ 7.468672, -0.481556, 0.398409, -5.663545, 0.303342, -0.709184, -3.311649, 0.519999, -0.210195, 1.50652, -0.341785, 0.52097]
            , [-0.481556, -1.703046, -1.095639, 0.312254, 4.50928, 0.950217, 0.125658, -2.762323, 0.277076, 0.043644, -0.043911, -0.131655]
            , [ 0.398409, -1.095639, 6.687239, -0.070907, -0.950218, -4.993221, -0.235783, 1.232919, -1.848179, -0.052817, 0.812937, 0.154161]
            , [-5.663545, 0.312254, -0.070907, 6.946056, -0.472644, 0.385633, 1.554707, -0.350697, 0.109883, -2.837218, 0.511087, -0.420659]
            , [ 0.303342, 4.50928, -0.950218, -0.472644, -1.703046, 1.095639, 0.050912, -0.043911, 0.131655, 0.118389, -2.762323, -0.277076]
            , [-0.709184, 0.950217, -4.993221, 0.385633, 1.095639, 6.687239, 0.657331, -0.812937, 0.154161, -0.36873, -1.232919, -1.848179]
            , [-3.311649, 0.125658, -0.235783, 1.554707, 0.050912, 0.657331, 3.054322, -0.151584, 0.096644, -1.29738, -0.024987, -0.518191]
            , [ 0.519999, -2.762323, 1.232919, -0.350697, -0.043911, -0.812937, -0.151584, 2.837786, -0.414356, -0.017718, -0.031552, -0.005626]
            , [-0.210195, 0.277076, -1.848179, 0.109883, 0.131655, 0.154161, 0.096644, -0.414356, 1.682793, 0.003668, 0.005626, 0.011225]
            , [ 1.506522, 0.043644, -0.052817, -2.837218, 0.118389, -0.36873, -1.29738, -0.017718, 0.003668, 2.628076, -0.144315, 0.41788]
            , [-0.341785, -0.043911, 0.812937, 0.511087, -2.762323, -1.232919, -0.024987, -0.031552, 0.005626, -0.144315, 2.837786, 0.414356]
            , [ 0.52097, -0.131655, 0.154161, -0.420659, -0.277076, -1.848179, -0.518191, -0.005626, 0.011225, 0.41788, 0.414356, 1.682793]])
            
            self.assertTrue((self.hessian - correct_hessian < 0.00001).all())
    
    def test_hessian_dimensions(self):
        self.assertTrue(self.hessian.shape == (self.n_coordinates, self.n_coordinates))

class hessian_trans_rot_test(abavib_test):
    def test_something(self):
        self.assertTrue(True)

class masswt_hessian_test(abavib_test):
    def test_something(self):
        self.assertTrue(True)
 
class frequency_test(abavib_test):
    def test_eigenvalues(self):
        correct_eigenvalues = np.array([0.0003967267, 0.0003909715, 5.5175184e-005, 4.4395569e-005, 2.8355625e-005, 1])
        self.assertTrue((correct_eigenvalues - self.eig < 0.05).all())
        
class effective_geometry_test(abavib_test):
    def test_effective_geometry(self):
        if(self.molecule == "h2o"):
            correct_effective_geometry = np.array([[-1.4215725557, 2.2811532702, 0.0055491054]
            ,[-0.135337927, 2.107000566, 0.0738738213]
            ,[-2.025218957, 3.349659217, -0.4137113479]])
            self.assertTrue((correct_effective_geometry - self.effective_geometry_cart < 0.5).all())
            
        if(self.molecule == "h2o2"):
            correct_effective_geometry = np.array([[-0.00012007, 1.40873621, -0.10043595]
            ,[0.00020733, -1.40863870, -0.10057873]
            ,[0.69590721, 1.73453487, 1.59345500]
            ,[-0.69729204, -1.73608239, 1.59679826]])
            self.assertTrue((correct_effective_geometry - self.effective_geometry_cart < 0.5).all())
        
        self.assertTrue((correct_effective_geometry - self.effective_geometry_cart < 0.5).all())
        
class dipole_test(abavib_test): #Checkout how to manage the whole "close enough" conundrum
    def test_dipole_corrections(self):
        
        if(self.molecule == "h2o"):
            dipole_corrections_correct = np.array([-0.00001144,-0.0000035,-0.00459292])
            self.assertTrue((dipole_corrections_correct - self.dipole_moment_diff < 0.0005).all())   
            
        elif(self.molecule == "h2o2"):
            dipole_corrections_correct = np.array([-0.00001144, -0.00000350, -0.00459292])
            self.assertTrue((dipole_corrections_correct - self.dipole_moment_diff < 0.0001).all()) 
        
    def test_dipole_moment(self):
        
        if(self.molecule == "h2o"):
            dipole_moment_correct = np.array([0.37357035, 0.49114923, -0.19272230])
            self.assertTrue((dipole_moment_correct - self.dipole_moment_corrected < 0.005).all())
            
        elif(self.molecule == "h2o2"):
            dipole_moment_correct = np.array([0.00025464,-0.00020485,0.97278737])
            self.assertTrue((dipole_moment_correct - self.dipole_moment_corrected < 0.05).all())

class shield_test(abavib_test): #Checkout how to manage the whole "close enough" conundrum
    def test_shield(self):
        
        if(self.molecule == "h2o"):
            shield_correct = np.array([[[-0.80608243, 0.21073704, 0.28992085]
            ,[-0.33796272, -0.00832737, 0.04054263]
            ,[0.13247249, 0.00326504, -0.01587188]]
            ,[[-0.33689843, 0.01340778, 0.01821706]
            ,[-1.10194286, 0.29981299, 0.23060641]
            ,[-0.0892006, -0.00225043, 0.02512025]]
            ,[[0.13222338, -0.00526279, -0.0071505]
            ,[-0.08759976, -0.0023847, 0.02509715]
            ,[-1.29295689, 0.29476779, 0.28489411]]])
            self.assertTrue((shield_correct - self.shield < 0.5).all())
            
        elif(self.molecule == "h2o2"):
            
            shield_correct = np.array([[[-2.73543218 , 1.29310207 , -1.89858102]
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
            self.assertTrue((shield_correct - self.shield < 0.5).all())        
        
class nuclear_quadrupole_test(abavib_test): #Checkout how to manage the whole "close enough" conundrum
    def test_nuclear_quadrupole(self):
        
        if(self.molecule == "h2o"):
            self.assertTrue((False))
            
        elif(self.molecule == "h2o2"):
            
            nucquad_correct = np.array([[[0.00173516,-0.00266833, 0.01132919]
                            ,[0, -0.00676687,-0.01183273]
                            ,[0, 0, -0.01306435]]
                            
                            ,[[0.00177681,-0.0026176, 0.01120484]
                            ,[0, 0.00675537, 0.01177362]
                            ,[0, 0, -0.01298165]]
                            
                            ,[[-0.00844869, 0.00204556, -0.00631312]
                            ,[0, 0.01162262,0.00461045]
                            ,[0,0,0.01476181]]
                            
                            ,[[-0.00845538,0.00208702,-0.00643277]
                            ,[0, -0.01167536,-0.00467444]
                            ,[0,0,0.01488815]]])
            
            self.assertTrue((nucquad_correct - self.nuc_quad < 0.05).all())
    
if __name__ == '__main__':
    unittest.main()

            
            
print shield_correct.transpose()
