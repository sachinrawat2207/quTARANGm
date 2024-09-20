"""
MIT License

Copyright (c) 2024 Sachin Singh Rawat, Shawan Kumar Jha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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