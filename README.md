# T-Nalg
Tensor Netwok Algorithms for EVERYONE
(It would be appriciated if you cite our tensor network review paper: https://arxiv.org/abs/1708.09213)

## DMRG for any finite-size lattices
### Two ways of using DMRG:
  1. Set up Parameters.py and run dmrg_finite_size in DMRG_anyH.py
  2. Run EasyStartDMRG directly
*The aim of EasyStartDMRG is to let everyone be able to use DMRG to simulate the ground state of any quantum models. One can run 'EasyStartDMRG' to start. 
========= For using EasyStartDMRG (v2018.06-1) ========= 
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
    3.The results including the MPS will be save in '.\data_dmrg' ===================================================================== 
* Before using, you need to install numpy, scipy, and matplotlib in your python
