# T-Nalg
Tensor Netwok Algorithms for EVERYONE
*If our code helps your project, please cite our tensor network review paper: S. J. Ran, arXiv.1708.09213 (https://arxiv.org/abs/1708.09213)
* Before using, you need to install numpy, scipy, and matplotlib in your python

## DMRG for any finite-size lattices
### Two ways of using DMRG:
  1. Set up Parameters.py and run dmrg_finite_size in DMRG_anyH.py
  2. Run EasyStartDMRG directly
*The aim of EasyStartDMRG is to let everyone be able to use DMRG to simulate the ground state of any quantum models. One can run 'EasyStartDMRG' to start. 
### To use EasyStartDMRG, you only need to know three things: 
    1.What you are simulating (e.g., Heisenber model, entanglement, ect.) 
    2.How to run a Python code 
    3.English 
    * It is ok if you may not know how DMRG works 
### Steps to use EasyStartDMRG: 
    1.Run 'EasyStartDMRG' 
    2.Input the parameters by following the instructions 
    3.Choose the quantities you are interested in 
### Some notes: 
    1.Your parameters are saved in '.\para_dmrg\_para.pr' 
    2.To read *.pr, use function 'load_pr' in 'Basic_functions_SJR.py' 
    3.The results including the MPS will be save in '.\data_dmrg' 
### Models that can be computed:
    1. Spin-1/2 models on any finite-size lattices
    2. Bosonic models on any finite-size lattices (need to munually perpare Parameters.py)
### Physics that can be computed:
    1. Ground states in MPS form
    2. Ground-state average of any one-body operator
    3. Ground-state average of any two-body operator
## Contents to be added soon
  1. Higher-spin models
  2. iDMRG [1] and iTEBD [2] for the ground states of infinite spin chains
  3. LTRG [3] for the thermodynamics of infinite spin chains
  4. Simple update [4] for the ground states of infinite 2D spin models
  5. ODTNS [5] for the thermodynamics of infinite 2D spin models
  6. AOP [6] for the ground states of 1D, 2D, 3D infinite models
  7. AOP [7] for the thermodynamics of 1D, 2D, 3D infinite models
## References
 [0] Our TN review: Shi-Ju Ran, Emanuele Tirrito, Cheng Peng, Xi Chen, Gang Su, Maciej Lewenstein. Review of Tensor Network Contraction Approaches, arXiv:1708.09213.   
 [1] iDMRG: Steven R. White. Density matrix formulation for quantum renormalization groups. Phys. Rev. Lett. 69, 2863 (1992).  
 [2] iTEBD: Guifre Vidal. Classical Simulation of Infinite-Size Quantum Lattice Systems in One Spatial Dimension. Phys. Rev. Lett. 98, 070201 (2007).   
 [3] LTRG: Wei Li, Shi-Ju Ran, Shou-Shu Gong, Yang Zhao, Bin Xi, Fei Ye, and Gang Su. Linearized tensor renormalization group algorithm for the calculation of thermodynamic properties of quantum lattice models. Phys. Rev. Lett. 106, 127202 (2011).  
 [4] Simple update: Hong-Chen Jiang, Zheng-Yu Weng, and Tao Xiang. Accurate determination of tensor network state of quantum lattice models in two dimensions. Phys. Rev. Lett. 101, 090603 (2008).  
 [5] ODTNS: Shi-Ju Ran, Wei Li, Bin Xi, Zhe Zhang, and Gang Su. Optimized decimation of tensor networks with super-orthogonalization for two-dimensional quantum lattice models. Phys. Rev. B 86, 134429 (2012).  
 [6] AOP (1D): Shi-Ju Ran. Ab initio optimization principle for the ground states of translationally invariant strongly-correlated quantum lattice models. Phys. Rev. E 93, 053310 (2016).  
 [7] AOP (2D and 3D): Shi-Ju Ran, Angelo Piga, Cheng Peng, Gang Su, and Maciej Lewenstein. Few-body systems capture many-body physics: Tensor network approach. Phys. Rev. B 96, 155120 (2017).
