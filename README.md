[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


**quTARANG** is a Python package designed for studying turbulence in quantum systems, specifically in atomic Bose-Einstein condensates (BECs). It utilizes the mean-field Gross-Pitaevskii equation (GPE) to simulate the dynamics of the condensates. The non-dimensional GPE implemented in quTARANG is given by

$$
i\partial_t\psi(\vec{r},t) = -\frac{1}{2}\nabla^2\psi(\vec{r},t) + V(\vec{r},t)\psi(\vec{r},t) + g|\psi(\vec{r},t)|^2\psi(\vec{r},t),
$$

where $\psi(\vec{r},t)$ is the macroscopic, non-dimensionalized complex wave function, $V(\vec{r},t)$ is the non-dimensionalized external potential, and $g$ is the coefficient of non-linearity, governing the strenth of interactions within the system.

This package is hardware-agnostic, allowing users to run simulations on either a CPU or a GPU by simply changing a flag. **quTARANG** uses the Time-Splitting Pseudo-Spectral (TSSP) method for evolving the system, ensuring both efficiency and accuracy. Additionally, the package can compute stationary states by evolving the GPE in imaginary time. It is equipped with the functions to compute statistical quantities like spectra and fluxes. It can compute the energy spectra using a conventional binning method, and a resolved spectra using the angle-averaged Wiener-Khinchin approach [see](https://journals.aps.org/pra/pdf/10.1103/PhysRevA.106.043322). 

## Directory structure of quTARANG
The directory structure of **quTARANG** package is as follows:
```
├── quTARANG
    ├── config
    ├── initial_cond
    ├── util
    ├── src
├── examples
├── para.py
├── main.py
└── postprocessing
```
- `quTARANG` directory contains the quTARANG's source files.
- `examples` directory contains the working examples of different 2-D and 3-D cases for both stationary state computation and dynamical evolution of system.
- `para.py` contains the different run parameters that needed to be set in order to perform a sumulaition.
- `main.py` is the file that should be executed to start the simulation and is used to define the initial conditions for some cases.

- `postprocessing` directory contains libraries and files used for data postprocessing, which includes:
    1. Computation and plotting of the spectra (compressible kinetic energies, incompressible kinetic energy and particle number) using a conventional binning method, along with a resolved spectra using the angle-averaged Wiener-Khinchin approach.
    2. Computation and plotting of fluxes for compressible kinetic energy, incompressible kinetic energy and particle number.
    3. Ploting the time series of energies as well as the the root mean square (RMS) values of the condensate.

## Packages required to run quTARANG
The following Python packages must be installed to run quTARANG and postprocess data using it.

    * `numpy` : To run the code on a CPU,
    * `cupy` : To run the code on a GPU,
    * `h5py` : To save the output in HDF5 format,
    * `matplotlib` : To plot the data,
    * `pyfftw`,
    * `imageio` : To generate animation.

In addition to the above packages, the user can also install LaTeX (preferably TeX-Live) to generate symbols and numbers in the LaTex style while generating plots using quTARANG.

## Running quTARANG
In order to run quTARANG, user needs to configure the parameters and set the initial conditions. This is done using the para.py and main.py files, respectively. Some working examples are provided in the `example` directory, a users can copy the contents of the `main.py` and `para.py` files from the `example` directory to  `main.py` and `para.py` files of `quTARANG`'s directory. Once the `para.py` and `main.py` file sets, he/she can perform a simulation using the followng command:
```python
python3 main.py
```
 
## Description of `para.py` and `main.py` file.

The parameters within the `para.py` file are described as follows:

| Parameters | Description | Values |
|------------|-------------|--------|
| `real_dtype` | Sets the precision of real arrays used in the code. | `"float32"`: Single precision. <br> `"float64"`: Double precision. |
| `complex_dtype` | Sets the precision of complex arrays used in the code. | `"complex32"`: Single precision. <br> `"complex64"`: Double precision. |
| `device` | Sets the device on which the code will run. | `"cpu"`: Run the code on a CPU. <br> `"gpu"`: Run the code on a GPU. |
| `device_rank` | Sets which GPU to use in multi-GPU systems, with values ranging from 0 to `(number of GPUs - 1)`. Default is 0. Remains ineffective for `device = "cpu"`. | `0` to `(number of GPUs - 1)`. |
| `Nx, Ny, Nz` | Sets the grid sizes along the $x$-, $y$-, and $z$-axes. | Set `Ny=1`, `Nz=1` for 1-D and `Nz=1` for 2-D simulations. 
| `Lx, Ly, Lz` | Sets the box lengths along the $x$-, $y$-, and $z$-axes. | Set `Ly=1`, `Lz=1` for 1-D and `Lz=1` for 2-D simulations. |
| `tmax, dt` | `tmax` sets the total simulation time, while `dt` determines the time step size. | |
| `g` | Sets the value of the nonlinearity parameter for the system. | |
| `inp_type` | This parameter determines how the initial condition will be set in the quTARANG. | `"fun"`: The initial condition is defined through a function (to be specified in `main.py`). <br> `"dat"`: Initial condition is provided through the two input wavefunction and potential files of `HDF5` file type. <br> `"pdf"`: The initial condition is set using predefined functions within the code. |
|`typ`|When `inp_type="pdf"`, this parameter sets the type of initial condition. Remains ineffective for other values of `inp_type`.|2D: `"rp"`, `"rv"`, and `"vl"` corresponds to smooth random phase, random vortices, and vortex lattice initial conditions, respectively. <br> 3D: `"rp"` corresponds to the smooth random phase initial condition|
| `in_path` | Sets the directory containing the initial wavefunction and potential files when `inp_type = "dat"`. Remains ineffective for other value of `inp_type`. | `"path/to/input/directory"` |
| `op_path` | Sets the output directory, where simulation data generated by quTARANG will be saved. | `"path/to/output/directory"` |
| `scheme` | Sets the numerical scheme. In current version, only TSSP is supported, so keep it unchanged. | `"TSSP"` |
| `imgtime` |  Sets whether the code computes the stationary state or evolves the system. | `True` (compute stationary state), `False` (evolve the system). |
| `delta` | Stopping criteria for stationary state computation (`imgtime = True`). It is the absolute difference in energy between consecutive steps, i.e., $\delta = \|E_n - E_{n-1}\|$. Remains ineffective for `imgtime = False`. | |
| `overwrite` | Prevents overwriting of the data already present inside the output directory. | `True` (overwrite), `False` (do not overwrite). |
|`save_wfc`|Sets whether the wavefunction will be saved or not |`True` (save), `False` (do not save).|
| `wfc_start_step`, `wfc_iter_step` | `wfc_start_step`: The number of iterations after which the wavefunction starts saving. `wfc_iter_step`: The interval between subsequent wavefunction saves. Remains ineffective for `save_wfc = False`. | |
| `save_rms` | Sets whether to save the time series of the root mean square (RMS) value of the condensate. | `True` (save), `False` (do not save). |
| `rms_start_step, rms_iter_step` | Controls RMS saving behavior. Similar to `wfc_start_step, wfc_iter_step`. | |
| `save_en` | Sets whether to save the time series of energy values. | `True` (save), `False` (do not save). |
| `en_start_step, en_iter_step` | Controls energy saving behavior. Similar to `wfc_start_step, wfc_iter_step`. | |
| `t_print_step` | Sets the intervals after which  the data will print on the terminal. | |

### Setting initial conditon
Based on the type of input for initial condition (`inp_type="fun"`, `inp_type="dat"` or `inp_type="pdf"`), the following are the possible cases for setting the intial conditions: 
#### Case I: Setting the initial condition by using function 
For `inp_type = "fun"` in `para.py`,
the user has to define the functions that will return the initial wavefunction and potential in `main.py`. These functions can be defined using the following aliases:

- `ncp`: An alias of NumPy(for a CPU) or cuPy(for a GPU), depending on the device used for code execution.
- `grid`: An alias of quTARANG's grid library containing the grids for the $x-, \ y-$, and $z-$ axes, and can be accessed using `x_mesh`, `y_mesh`, and `z_mesh` variables, respectively.

The following is the example showing `main.py` file, where the functions are defined for wavefunction, $\psi(\vec{r},0)=\left(\frac{1}{\sqrt{2}\pi}\right)^{1/2} e^{-(x^2+2y^2)/4}$ and potential, $V(\vec{r},0)=\frac{1}{2}(x^2+4y^2)$.

```python
#main.py
from quTARANG.src.lib import gpe
from quTARANG.src import evolution
from quTARANG.src.univ import grid, fns
from quTARANG.config.config import ncp
import quTARANG.config.mpara as para

##########################################################################
V = 0.5*(grid.x_mesh**2 + 4*grid.y_mesh**2)
def wfcfn():
    return 1/ncp.sqrt(ncp.sqrt(2)*ncp.pi)*ncp.exp(-(grid.x_mesh**2 + 2*grid.y_mesh**2)/4)

def potfn(t):
    return V + 0*t

G = gpe.GPE(wfcfn = wfcfn, potfn = potfn)
##########################################################################

evolution.time_advance(G)
```
The potential function can be time dependent so, always takes a input parameter `t` in the definition even if the potential is time independent. The user can defines time dependent potential fuction using that `t`. Once the functions are defined the user needs to pass references of those functons to `G`, a instance of the `gpe` class as `G = gpe.GPE(wfcfn = wfc_func, potfn = pot_func)`, where `wfc_func` and `pot_func` are reference of defined fucntions.

#### Case II: Setting initial condtion using predefined initial conditoin or  by passing the path of directory containing wavefunction and potential file in `HDF5` format.
For both `inp_type="dat"` and `inp_type="pdf"` in `para.py`, the `main.py` will remain same. When `inp_type = "dat"`, the user have to set `inp_path` variable of `para.py` to the path of directory having the wavefunction and potential input data files in a form of two HDF5 files: (1) `wfc.h5`, which should have a dataset named `wfc` containing wavefunction data, and (2)`pot.h5`, which should have a dataset named `pot` containing potential data. 
While for `inp_type = "pdf"`, the user has to set the `typ` variable.

The following is the `main.py` file corresponding to this case. 
```python
#main.py
from quTARANG.src.lib import gpe
from quTARANG.src import evolution
from quTARANG.src.univ import grid, fns
from quTARANG.config.config import ncp
import quTARANG.config.mpara as para

##########################################################################
G = gpe.GPE()
##########################################################################

evolution.time_advance(G)
```


## Ouputs
When the simulation completes successfully, the output files generated during runtime will be saved inside the output directory. These files are stored in `HDF5` format. In case of the dynamical evolution (`imgtime = False` in `para.py`), the following directories/files are generated inside the output directory.:
- `wfc` : The `wfc` directory stores wavefunctions at different points in time. The filenames follow the format `wfc_<time>.h5`, where `<time>` represents the simulation time at which the wavefunction was generated. For example, `wfc_10.000000.h5` indicates that the wavefunction was generated at time $t = 10$. In each of the wavefucntion file, the data is saved in the `wfc` dataset which can be easily accessed by Python's `h5py` library. 
- `pot.h5` : This file contains the data for the potential at `t=0` and in this file data for potential is saved in `pot` dataset. 
- `energies.h5` : This file contains the time series of different types of energies.
- `rms.h5` : This files contains the time series of RMS values: $x_{rms}, \ y_{rms}$, $r_{rms}$ for a 2-D run, and $x_{rms}, \ y_{rms} \ z_{rms}$, $r_{rms}$ for a 3-D run.

- `para.py` and `main.py`: These are copies of the original parameter and main files (`para.py` and `main.py`) used at the time of the simulation. These files allow user to check the initial conditions and parameters used for the simulation.

In case of computing the stationary state (`imgtime = True` in `para.py`), a wavefunction directory containg the wavefunctions at different iterations and after the end of the iterations will generate. If `save_wfc=False` in `para.py` file, it will only generate one wavefuncton file at the end of iterations. The wavefuntions filenames follow the format `wfc_<iterations>.h5`, where `<iterations>` represents the no of iterations after which the wavefunction was saved. For example, `wfc_1000.h5` indicates that the wavefunction was saved at the end of 1000th iteration. The wavefuntion whoose iteration number is largest among all the wavefucntion will be the wavefuction corresponding to the ground state of the system. In each of the  wavefucntion file, the data will stored in the `wfc` dataset.

## Postprocessing 
Once a simulations using ***quTARANG*** has completed successfully, the data generated in the output directory can be postporocessed using the files within the `postprocessing` directory. The structure of the directories and files within the this directory are as follows:

```
├── src
├── op_path.py    
├── plot_energy.py
├── plot_rms.py
├── plot_animation.ipynb
└── plot_spectra.ipynb
``` 

The `src` directory contains the classes and functions for the computation of spectra, fluxes and generation of animation.The user need not the change the content of the src file. 

To perform postprocessing on the data, the user has to simply set the path of the data which wase generated by quTARANG 
 in `op_path.py` file. Once path has set, a user can plot the time series for energy and RMS by simply running `plot_energy.py` and `plot_rms.py` files, respectively. The plots and anmations correspoending to the density and phase will be generated using the jupyter notebook named `plot_animation.ipynb` while the spectra and flux plots can be generated using the another jupyter notebook named `plot_spectra.ipynb`. The comments on the cells of these notebooks explains the usage of the code inside those cells. The plots generated will be saved inside a newly created subdirectory named `postprocessing` within the output directory.

## Test cases

1. 2D case:  

    $$\psi(\vec{r},0)=\left(\frac{1}{\sqrt{2}\pi}\right)^{1/2} e^{-(x^2+2y^2)/4}$$

    $$V(\vec{r},0)=\frac{1}{2}(x^2+4y^2)$$ 

    The corresponding `main.py` and `para.py` files for this case are as follows:

    ```python
    #main.py
    from quTARANG.src.lib import gpe
    from quTARANG.src import evolution
    from quTARANG.src.univ import grid, fns
    from quTARANG.config.config import ncp
    import quTARANG.config.mpara as para

    ##########################################################################
    V = 0.5*(grid.x_mesh**2 + 4*grid.y_mesh**2)
    def evolve_wfc2d():
        return 1/ncp.sqrt(ncp.sqrt(2)*ncp.pi)*ncp.exp(-(grid.x_mesh**2 + 2*grid.y_mesh**2)/4)

    def evolve_pot2d(t):
        return V + 0*t

    G = gpe.GPE(wfcfn = evolve_wfc2d, potfn = evolve_pot2d)
    ##########################################################################

    evolution.time_advance(G)
    ```

    ```python
    #para.py
    #=======================================#================================================================================
    #                       Change the following parameters
    #================================================================================
    real_dtype = "float64"
    complex_dtype = "complex128"

    pi = 3.141592653589793

    # Device Setting 
    device = "gpu"             # Choose the device <"cpu"> to run on cpu and gpu to run on <"gpu">
    device_rank = 1            # Set GPU no in case if you are running on a single GPU else leave it as it is

    # Set grid size 
    Nx = 256
    Ny = 256
    Nz = 1
        
    # Set box length
    Lx = 16
    Ly = 16
    Lz = 1

    # Set maximum time and dt
    tmax = 8    
    dt = 0.001

    # Choose the value of the non linerarity
    g = 2

    inp_type = "fun"       # Choose the initial condition type among <"fun">, <"dat"> and <"pdf">

    typ = "rp"            # In case of inp_type = "pdf" set the type of initial condition <"rp">, <"rv">, <"vl"> for 2D and <"rp"> for 3D.

    # If inp_type = "dat" then set the input path
    in_path = "/path/to/input_directory"


    # Set output folder path
    op_path = "../output_evolve2D"

    scheme = "TSSP"         

    imgtime = False          # set <False> for real time evolution and <True> for imaginary time evolution
    delta = 1e-12

    overwrite = False

    # Wavefunction save setting
    save_wfc = True
    wfc_start_step = 0
    wfc_iter_step = 500

    # Rms save setting
    save_rms = True
    rms_start_step = 0
    rms_iter_step = 10

    # Energy save setting
    save_en = True
    en_start_step = 0
    en_iter_step = 100

    # Printing iteration step
    t_print_step = 1000


    ```
    On the successful execution of the code, a user can postprocess the data by first setting the `path` variable in the `op_path.py` file located in the `postprocessing` directory, after which the user can perform the analysis.

    The following plots shows the time series of the RMS of the condedsates and energy generated by running `plot_rms.py` and `plot_energy.py`:

    | ![Image 1](output_data/2D/rms.jpeg) | ![Image 2](output_data/2D/energy.jpeg) |
    |:---------------------:|:----------------------:|

2. 3D case:

    $$\psi(\vec{r},0)=\left(\frac{8}{\pi}\right)^{3/4}\text{exp}(-2x^2-4y^2-8z^2)$$

    $$V(\vec{r},0)=\frac{1}{2}(x^2+4y^2+16z^2)$$

    The corresponding `main.py` and `para.py` files for this case are as follows: 

    ```python
    #para.py
    #================================================================================
    #                       Change the following parameters
    #================================================================================
    real_dtype = "float64"
    complex_dtype = "complex128"

    pi = 3.141592653589793

    # Device Setting 
    device = "gpu"             # Choose the device <"cpu"> to run on cpu and gpu to run on <"gpu">
    device_rank = 0            # Set GPU no in case if you are running on a single GPU else leave it as it is

    # Set grid size 
    Nx = 256
    Ny = 256
    Nz = 256
        
    # Set box length
    Lx = 16
    Ly = 16
    Lz = 16

    # Set maximum time and dt
    tmax = 5    
    dt = 0.001

    # Choose the value of the non linerarity
    g = 0.1

    inp_type = "fun"        # Choose the initial condition type among <"fun">, <"dat"> and <"pdf">

    typ = "rp"            # In case of inp_type = "pdf" set the type of initial condition <"rp">, <"rv">, <"vl"> for 2D and <"rp"> for 3D.

    # If inp_type = "dat" then set the input path
    in_path = "/path/to/input_directory"

    # Set output folder path
    op_path = "../output_evolve3D"


    imgtime = False          # set <False> for real time evolution and <True> for imaginary time evolution
    delta = 1e-12

    overwrite = False

    # Wavefunction save setting
    save_wfc = True
    wfc_start_step = 0
    wfc_iter_step = 500

    # Rms save setting
    save_rms = True
    rms_start_step = 0
    rms_iter_step = 10

    # Energy save setting
    save_en = True
    en_start_step = 0
    en_iter_step = 100

    # Printing iteration step
    t_print_step = 1000

    ```
    ```python
    #main.py
    from quTARANG.src.lib import gpe
    from quTARANG.src import evolution
    from quTARANG.src.univ import grid, fns
    from quTARANG.config.config import ncp
    import quTARANG.config.mpara as para

    ##########################################################################
    V = 0.5*(grid.x_mesh**2 + 4 * grid.y_mesh**2 + 16 * grid.z_mesh**2)
    def evolve_wfc3d():
        return (8/ncp.pi)**(3/4)*ncp.exp(-2*(grid.x_mesh**2 + 2 * grid.y_mesh**2 + 4 *grid.z_mesh**2))

    def evolve_pot3d(t):
        return V + 0*t

    G = gpe.GPE(wfcfn = evolve_wfc3d, potfn = evolve_pot3d)
    ##########################################################################

    evolution.time_advance(G)
    ```

    The following plots shows the time series of the RMS of the condedsates and energy generated by running `plot_rms.py` and `plot_energy.py`:

    | ![Image 1](output_data/3D/rms.jpeg) | ![Image 2](output_data/3D/energy.jpeg) |
    |:---------------------:|:----------------------:|

