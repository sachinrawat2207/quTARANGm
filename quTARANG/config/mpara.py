from pathlib import Path
import sys 
para_dir = (str(Path(__file__).resolve().parents[1]))
sys.path.append(str(para_dir))
from para import *

if Nx != 1 and Ny == 1 and Nz == 1:
    dimension = 1

elif Nx != 1 and Ny != 1 and Nz == 1:
    dimension = 2

elif Nx != 1 and Ny != 1 and Nz != 1:
    dimension = 3  
    
# Used for computation of time
t0_c = 0
ti_c = 0
tf_c = 0