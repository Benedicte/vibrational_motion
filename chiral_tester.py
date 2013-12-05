import unittest
import abavib as av
import read_input as ri
import numpy as np

molecule = "fluoromethane"
input_name = "input_" + molecule + "/"
output_file_name = "output/" + molecule
open(output_file_name, 'w').close() # As we are appending to the output, the old results must be deleted before each run

class abavib_test(unittest.TestCase):
    
    def setUp(self):
    
    #The reason we use this one, is because there are any number of eigenvectors which are correct eigenvectors, for the purpose of testing
    #we use the same one that DALTON operates with
    
        self.molecule = "fluoromethane"
            
        if(self.molecule == "fluoromethane"):
            self.eig = np.array([0.000156325, 0.000052708, 0.000083924, 0.000087366, 0.000049731, 0.000036663, 0.000105884, 0.000086732, 0.000035581])
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
            #self.eig1, self.eigvec1, self.freq, self.eigvec_full1 = \
            #    av.fundamental_freq(self.hessian, self.num_atoms_list, \
            #    self.charge_list, self.coordinates, self.n_atoms, self.masses)#Check out the 1s i made
            self.cubic_force_field = ri.read_cubic_force_field(self.cff_name,#Remember to switch to av. for h2o\  
             self.n_coordinates) 
            #self.cff_norm, self.cff_norm_reduced = av.to_normal_coordinates_3D(self.cubic_force_field, self.eigvec_full, self.n_atoms)
            #effective_geometry_norm = av.effective_geometry(self.cff_norm_reduced, self.freq, self.n_atoms)
            #self.effective_geometry_cart = av.to_cartessian_coordinates(effective_geometry_norm, self.n_atoms, self.eigvec)
            
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
 
class frequency_test(abavib_test):

    def test_frequencies(self):
        if(self.molecule == "h2o2"):    
            correct_frequency = np.array([0.0570, 0.0435, 0.0413, 0.0343, 0.0294, 0.0168, 0,0,0,0,0,0])
            self.assertTrue(np.allclose(correct_frequency, self.freq, rtol=0.02, atol=0.0003)) 

        if(self.molecule == "h2o"):    
            correct_frequency = np.array([0.037739, 0.045369, 0.025234, 0,0,0,0,0,0])

            self.assertTrue(np.allclose(correct_frequency, self.freq, rtol=0.02, atol=0.0003))       
    
    def test_eigvec(self):
        if(self.molecule == "h2o2"):
            correct_eigvec = np.array([[ -0.00131353, -0.00001741, 0.00029587, -0.00016271, 0.00000038, 0.00006501]
                        ,[ 0.00007785, -0.00060863, -0.00084065, -0.00064259, -0.00032658, 0.00406074]
                        ,[ -0.00018153, 0.00151054, 0.00052282, -0.000208, -0.00088988, -0.00002127]
                        ,[ 0.00116367, 0.00047638, -0.00027149, -0.00042556, 0.00006439, -0.00006238]
                        ,[ 0.00000617, -0.0007995, 0.00059862, -0.0006688, 0.00040987, -0.00405949]
                        ,[ 0.00036161, -0.00151616, 0.00016393, -0.00002019, -0.00098291, -0.00002758]
                        ,[ 0.01633121, -0.00137943, 0.00208677, 0.00294594, 0.00196957, 0.00042724]
                        ,[ -0.00289218, 0.00884232, 0.01679361, 0.01035385, 0.00450935, 0.00316981]
                        ,[ 0.00144209, -0.00874834, -0.00591666, 0.01308062, 0.01362057, 0.00033726]
                        ,[ -0.01395275, -0.00590483, -0.00247371, 0.00639033, -0.0029974, -0.00046896]
                        ,[ 0.00155876, 0.01350575, -0.01295232, 0.01045877, -0.0058313, -0.00318957]
                        ,[ -0.00430002, 0.00883742, -0.0049825, -0.00945915, 0.01610197, 0.00043797]])
                        
            vfunc = np.vectorize(np.absolute)
            correct_eigvec = vfunc(correct_eigvec)
            self.eigvec = vfunc(self.eigvec)
        
            self.assertTrue(np.allclose(correct_eigvec, self.eigvec, rtol=0.02, atol=0.0003))
            
        if(self.molecule == "h2o"):
            
            h2o_eigvec = np.array([[0.003447, -0.039874, -0.067216]
            ,[-0.072965, 0.008106, -0.017140]
            ,[0.028630, -0.003180, 0.006726]

            ,[-0.019459, 0.890699, 0.354563]
            ,[0.351730, -0.268710, 0.458153]
            ,[-0.138013, 0.105431, -0.179775]

            ,[-0.035255, -0.257865, 0.712205]
            ,[0.806271, 0.140066, -0.186130]
            ,[-0.316368, -0.054958, 0.073029]])
        
            self.assertTrue(np.allclose(h2o_eigvec, self.eigvec, rtol=0.02, atol=0.0003))
            
class cubic_force_field_test(abavib_test):
    def test_cff(self):
        self.assertTrue(True)
        
class optical_rotation_test(abavib_test): 

    def setUp(self):
        super(optical_rotation_test, self).setUp()
        optrot_deriv = ri.read_optrot(self.input_name + "OPTROT", self.n_nm)    
        self.uncorrected_values, self.values_correction, self.corrected_values = ri.read_DALTON_values_3d_reduced(self.input_name + "OPTROT")
        self.optrot_correction, self.optrot = av.get_3D_property("OPTROT", optrot_deriv, self.uncorrected_values, self.n_nm, self.eig) 
        print optrot_deriv
        print self.optrot
        print self.values_correction
               
    def test_optical_rotation_corrections(self):
        
        self.assertTrue(np.allclose(self.values_correction, self.optrot_correction, rtol=0.03, atol=0.0003))
            
    def test_optical_rotation_values(self):
        ri.write_to_file(self.molecule, "Optical Rotation", self.optrot)
        self.assertTrue(np.allclose(self.corrected_values, self.optrot, rtol=0.01, atol=0)) 

            
if __name__ == '__main__':
    unittest.main()
