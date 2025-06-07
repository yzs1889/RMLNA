# RMLNA

## Requirements：
- Tensorflow 1.14.0
- Scikit-learn
- numpy
- keras
- lime
- mstraj
- msmbuilder
- xlrt
- XlsxWriter

## Data Preperation Note：
1. Conformational ensembles of different states extracted from the trajectories based on the density map should first remove all hydrogens to ensure accuracy for the model.
2. Conformations  of different states should have the same atoms and residues.


### Usage：
python main.py --nc1_file='state1.nc' --nc2_file='state2.nc' --pdb1_file='state1.pdb' --pdb2_file='state2.pdb' --print_acc=1 --save_models=1 --print_detail=1 --atom_file='atom-important.xlsx' --res_file='res-important.xlsx' 

