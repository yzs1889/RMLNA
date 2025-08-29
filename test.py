import os
import sys
import numpy as np
import math
from tensorflow.keras.models import load_model
import mdtraj as md
from sklearn import preprocessing
res_id = []
trs = [[2.0413690, -0.5649464, -0.3446944], 
       [-0.9692660, 1.8760108, 0.0415560],
       [0.0134474, -0.1183897, 1.0154096]]

def read_traj(file_nc,file_top):               
    traj = md.load(file_nc,top=file_top)       
    return traj                                                                                  

def dis_bias(traj0, traj1):                                 
    traj_after = md.Trajectory.superpose(traj0+traj1, traj0+traj1, frame=0)
    traj0_aligned = traj_after[:len(traj0)]
    traj1_aligned = traj_after[len(traj0):]   
    return traj0_aligned, traj1_aligned 
                 
def xyz_to_rgb(xyz, trs=trs):
    # rgb = trs * xyz
    rgb = []
    for i in range(3):
        tmp = xyz[0] * trs[i][0] + xyz[1] * trs[i][1] + xyz[2] * trs[i][2]
        rgb.append(tmp)        
    return rgb

def sca_xyz(xyz, min=0, max=255):
    x = np.array(xyz)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(min,max))
    x_minmax = min_max_scaler.fit_transform(x)
    x_minmax2 = []
    for item in x_minmax:
        x_minmax2.append([int(item[0]),int(item[1]),int(item[2])])        
    return x_minmax2

def rgb_to_hex(rgb):
    string = ''
    digit = list(map(str, range(10))) + list("ABCDEF")
    for i in rgb:
        a1 = i // 16
        a2 = i % 16
        string += digit[a1] + digit[a2]
    return string

def traj_to_hex(traj):
    traj = traj.xyz   
    traj1 = []
    # xyz->rgb）
    for item in traj:
        item1 = []
        for atom in item:
            atom1 = xyz_to_rgb(atom)
            item1.append(atom1)
        traj1.append(item1)

    traj2 = []
    #（0,255）
    for item in traj1:
        item2 = sca_xyz(item, min=0, max=255)
        traj2.append(item2)
    pixel_map = traj2
    
    traj_hex = []
    for item in traj2:
        tep = []
        for atom in item:
            atom1 = rgb_to_hex(tuple(atom))
            rgb_dec = int(atom1.upper(), 16)
            tep.append([rgb_dec])
        traj_hex.append(tep)        
    return traj_hex, pixel_map

def load_traj(file0_1, file1_1, file0_2, file1_2):
    traj0_full = read_traj(file0_1, file0_2)                                     
    traj1_full = read_traj(file1_1, file1_2)      
                                                                           
    traj0_aligned, traj1_aligned = dis_bias(traj0_full, traj1_full)  
    residue_to_atom_indices = {}
    for i, residue in enumerate(traj1_aligned.topology.residues):
        residue_to_atom_indices[residue.resSeq] = [atom.index for atom in residue.atoms]

    import time
    start = time.time()
    n_frames = traj1_aligned.n_frames 
    traj1_hex, pixel_map1 = traj_to_hex(traj1_aligned)
    end = time.time()
    return traj1_hex, pixel_map1, residue_to_atom_indices, n_frames 

def traj_to_pic(traj1_hex, pixel1):
    # size*size
    atom_n = len(traj1_hex[0])
    import math
    size = math.ceil(atom_n**0.5)

    index_map = []
    for i in range(atom_n):
        row = i // size
        col = i % size
        index_map.append((row, col))    
      
    traj1_pic = []
    for item in range(len(traj1_hex)):
        for ti in range(size*size-atom_n):
            traj1_hex[item].append([0])
        pic = []
        for i in range(size):
            line = []
            for j in range(size):
                line.append(traj1_hex[item][i*size+j])
            pic.append(line)
        traj1_pic.append(pic)    
      
    pixel_map1 = []
    for item in range(len(pixel1)):
        for ti in range(size*size-atom_n):
            pixel1[item].append([0,0,0])
        pic = []
        for i in range(size):
            line = []
            for j in range(size):
                line.append(pixel1[item][i*size+j])
            pic.append(line)
        pixel_map1.append(pic)    
    return traj1_pic, pixel_map1, index_map, size
                                       
def modify_validation_set(X_test, res_id, residue_to_atoms, index_map, size):   
    modified_X_test = []
    retain_res_set = set(res_id)
    for img in X_test:
        img_copy = [[pixel.copy() for pixel in row] for row in img]
        for residue_id, atom_indices in residue_to_atoms.items():
            if residue_id not in retain_res_set:
                for atom_idx in atom_indices:
                    row, col = index_map[atom_idx]
                    img_copy[row][col] = [0, 0, 0]
        modified_X_test.append(img_copy)
    return modified_X_test

def data_encode(X_test):
    X_test = np.array(X_test)
    X_test = X_test.astype('float32')
    X_test /= 255
    return X_test

def eva_cross_acc_all(models, X_encode, y_test):
    total_acc = 0.0
    for i, model in enumerate(models):
        y_test_pre = model.predict_classes(X_encode)
        acc_test = com_accuracy_cross(y_test, y_test_pre)
        total_acc += acc_test
    return total_acc
    
def com_accuracy_cross(y1,y2):
    n = 0
    length_y = len(y1)    
    for i in range(length_y):
        if y1[i]==y2[i]:
            n += 1           
    accuracy = n/length_y
    return accuracy

if __name__ == "__main__":
    file0_1 = sys.argv[1]  # nc1
    file0_2 = sys.argv[2]  # pdb1
    file1_1 = sys.argv[3]  # nc2
    file1_2 = sys.argv[4]  # pdb2
    nc2_type = sys.argv[5] # label        
                             
    traj1_hex, pixel1, residue_to_atom_indices, n_frames = load_traj(file0_1, file1_1, file0_2, file1_2)
    traj1_pic, pixel_map1, index_map, size = traj_to_pic(traj1_hex, pixel1)   
    modified_X_test = modify_validation_set(pixel_map1, res_id, residue_to_atom_indices, index_map, size)                                             
    X_encode = data_encode(modified_X_test)
    y_test = [int(nc2_type) for i in range(n_frames)]
    
    model_paths = ['model{}.h5'.format(i) for i in range(5)]
    models = []
    for path in model_paths:
        model = load_model(path)
        models.append(model) 
        
    acc_total = eva_cross_acc_all(models, X_encode, y_test)
    acc_average = acc_total / len(models)
    print("\nAverage Accuracy: {:.4f}".format(acc_average))
    print("\n")
