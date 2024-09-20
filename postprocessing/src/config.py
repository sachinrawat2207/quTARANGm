import sys
import h5py as hp
from pathlib import Path 
import matplotlib.pyplot as plt 
import os 

############################################
sys.path.append(str(Path(__file__).resolve().parents[1]))
from op_path import *
############################################
device = 'cpu'
op_dir = Path(path)/'postprocessing/'

if not Path(op_dir).exists():
    os.makedirs(op_dir)

sys.path.insert(0, path)
import para 

if para.Nx != 1 and para.Ny == 1 and para.Nz == 1:
    dimension = 1

elif para.Nx != 1 and para.Ny != 1 and para.Nz == 1:
    dimension = 2

elif para.Nx != 1 and para.Ny != 1 and para.Nz != 1:
    dimension = 3