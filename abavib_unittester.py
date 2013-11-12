import unittest
import abavib as av
import numpy as np

class abavib_test(unittest.TestCase):
    def setUp(self):
        self.input_name = "input_h2o/"
        self.mol_name = self.input_name + 'MOLECULE.INP'
        self.coordinates, self.masses,  self.num_atoms_list \
          ,self.charge_list, self.n_atoms = av.read_molecule(self.mol_name)

class read_molecule_test(abavib_test):        
    def test_coordinates(self):
        self.correct_coordinates = [[-0.01754167, -1.37824361, -0.01186413],
                        [ 0.68558652, 1.69426236, 1.54662119],
                        [-0.66804485, -1.72386461, 1.50422431]]
        self.correct_coordinates = np.array(self.correct_coordinates)
        self.assertTrue((self.coordinates == self.correct_coordinates).all())
    
    def test_masses(self):
        self.correct_masses = [15.9994, 1.00794, 1.00794]
        self.assertSequenceEqual(self.masses, self.correct_masses)
        
    def test_num_atoms_list(self):
        self.assertSequenceEqual(self.num_atoms_list, [1,2])
        
    def test_charge_list(self):
        self.assertSequenceEqual(self.charge_list, [8,1])
        
    def test_n_atoms(self):
        self.assertEquals(self.n_atoms, 3)
        
class read_hessian_test(abavib_test):
    def test_something(self):
        self.assertTrue(False)

class hessian_trans_rot_test(abavib_test):
    def test_something(self):
        self.assertTrue(False)

class masswt_hessian_test(abavib_test):
    def test_something(self):
        self.assertTrue(False)

        
if __name__ == '__main__':
    unittest.main()

