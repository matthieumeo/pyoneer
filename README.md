# pyoneer
## Description
`Pyoneer` is a [Python 3](https://www.python.org)  package for the continuous recovery of non-bandlimited periodic signals with finite rates of innovation (e.g. Dirac streams) from generalised measurements. This package contains notably the reference implementation of the *Cadzow Plug-and-play Gradient Descent (CPGD)* algorithm, proposed by by Matthieu Simeoni, Adrien Besson, Paul Hurley and Martin Vetterli in the paper [Cadzow Plug-and-Play Gradient Descent for Generalised FRI]() [1].

The results of the paper  are provided in the folder `./results` and can be fully reproduced using the routines in the folder `./benchmarking`. 

>### Abstract of [1]
>The finite rate of innovation (FRI) framework allows perfect recovery of sum of sinusoids providing a sufficient number of uniform samples are measured. Introduced recently, Generalised FRI (GenFRI) extends FRI to cases where linear combinations of uniform samples are acquired. Signal reconstruc- tion is recast as a joint optimisation problem whose solution leads to both the Fourier-series coefficients of the signal of interest and the corresponding annihilating filter, two key ingredients for signal reconstruction within FRI. The algorithmic machinery set up to solve such a problem is computationally heavy and the solutions come with no guarantee of convergence. In this work, we propose to revisit signal reconstruction in GenFRI. Rather than solving a challenging joint estimation problem, signal reconstruction is recast as a non-convex optimisation problem on the Fourier-series coefficients. We introduce a projected gradient descent (PGD) method for signal reconstruction which is shown to converge to a local minimum for any sampling strategy but involves a proximal operator with no-closed form expression. We therefore propose an inexact PGD method, coined as Cadzow plug-and-play gradient descent (CPGD), where the proximal step involved in PGD is replaced by Cadzow denoising, a well known denoising algorithm in FRI. Based on recent results on plug-and- play algorithms, we provide local convergence guarantees for CPGD. Through an extensive number of numerical simulations, we demonstrate the superiority of CPGD against state-of-the-art algorithms in the case of non-uniform time sampling of streams of Dirac.

## Organisation of the `pyoneer` package
* `./benchmarking` contains the main routines for (re)-generating the results of the paper [1]:
    - The script `./benchmarking/reproduce_simulation_results.py` assesses the performances (positioning errors, execution times, number of iterations) of LS-Cadzow, CPGD and GenFRI for various oversampling parameters, peak signal-to-noise (PSNR) ratios and noise levels. 
    - The script `./benchmarking/reproduce_execution_times.py` compares the execution times of CPGD and GenFRI for various oversampling parameters.
* `./results` contains the results of the simulations (plots and pickle files).
* `./pyoneer/algorithms` contains the class definitions of the three reconstruction/denoising algorithms Cadzow, CPGD and GenFRI (see [1] for more details).
* `./pyoneer/model` contains routines for generating and sampling Dirac streams.
* `./pyoneer/operators` contains classes and routines for linear operators used in generalised FRI problems.
* `./pyoneer/plots` contains plotting routines as well as the custom matplotlib style file used to generate the plots of the paper. 
* `./pyoneer/utils` contains routines for FRI reconstruction as well as miscellaneous routines defining useful mathematical functions.

## Installation 
*Note: The following instructions have been tested on a MacBook Pro (16-inch, 2019) with macOS Catalina version 10.15.2 (19C57).*

`Pyoneer` is a Python 3 package relying on multiple open source packages written by third-partites (you can see all the import calls of pyoneer using the command `grep -h -R -e "^import" --include '*.py' ./ | sort -h | uniq`). To facilitate the management of cross-dependencies, we make use of [Miniconda](https://conda.io/miniconda.html), a light-weight version of the package and environment management open source software [Anaconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#anaconda-glossary). 

#### Installing and configuring Miniconda
If you are on macOS Catalina and had Miniconda or Anaconda installed prior to upgrading to the latest macOS, then your installation is probably broken due to new restrictions from Apple. To properly install Miniconda on macOS Catalina, download the binary file at [this link](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh). From the terminal, navigate to the location where the file was downloaded and execute 
```
bash Miniconda3-latest-MacOSX-x86_64.sh
```
Follow the prompts. Then, run `source ~/miniconda3/bin/activate` followed by `conda init zsh`. Finally, reopen the terminal and conclude the installation with `conda update`.


We are now ready to create a conda environment named `pyoneer` with Python 3.7 as the default python version:
```
conda create -n pyoneer python=3.7 
```
We will set the channel priority to strict, and add `conda-forge` as channel with higher priority.
```
conda config --set channel_priority strict
conda config --add channels conda-forge
```
These settings are stored in the hidden configuration file `~/.condarc`. Execute `cat ~/.condarc` to check that this is indeed the case.  

We can now install the relevant packages in the created environment:
```
conda install -n pyoneer -c conda-forge matplotlib numpy astropy h5py scipy joblib=0.13.0 numba pandas dask
```
We are done with the installation of dependencies. 
>Note: On versions of joblib `>0.13.0`, we experienced worse performances with the backend `multiprocessing`. We hence recommend sticking with version `0.13.0` for now.

#### Update `PYTHONPATH`
To use the custom `pyoneer` package, we need to tell to Python where to look for it. To this end, navigate with your terminal to the root of the repository and type:
```
echo "export PYTHONPATH=$PYTHONPATH:${PWD}" >> ~/.zshrc
```
Close and reopen you terminal for the changes to take place.

## Authors
*M. Simeoni and A. Besson have contributed equally to this work.*
* [Matthieu Simeoni](mailto:matthieu.simeoni@gmail.com), Laboratoire de Communications
Audiovisuelles (LCAV), École Polytechnique Fédérale de Lausanne (EPFL), CH-1015, Lausanne, Switzerland.
* [Adrien Besson](mailto:adribesson@gmail.com), E-Scopics, Saint-Cannat, France.
* [Paul Hurley](mailto:Paul.Hurley@westernsydney.edu.au), Western Sydney University (WSU), Australia.
* [Martin Vetterli](mailto:Paul.Hurley@westernsydney.edu.au), Laboratoire de Communications
Audiovisuelles (LCAV), École Polytechnique Fédérale de Lausanne (EPFL), CH-1015, Lausanne, Switzerland.

> *For questions related to this package please contact: matthieu.simeoni@gmail.com*
## License
>MIT License
>
>Copyright (c) 2020 Matthieu SIMEONI, Adrien BESSON, Paul HURLEY and Martin VETTERLI
>
>Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
>
>The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
>
>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

