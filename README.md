# RMLNA

### Step 1. Conformational states identification 
1. Select and calculate appropriate reaction coordinates for density maps.
2. For any two reaction coordinates, incorperating them as a *.dat file.
3. Run **density.ipynb**
   
### Step 2. Biased residues identification
1. Prepare data for the interpretable model：
   
   Conformational ensembles of different states are extracted from the trajectories based on the density map. All hydrogens of the conformations should be first removed to ensure accuracy for the model. Conformations of different states should have the same atoms and residues.
   
2. Set up environments for the model following the requirements below:
  - Tensorflow 1.14.0
  - Scikit-learn
  - numpy
  - keras
  - lime
  - mstraj
  - msmbuilder
  - xlrt
  - XlsxWriter
3. Train models and identify important residues：

   **python main.py --nc1_file='state1.nc' --nc2_file='state2.nc' --pdb1_file='state1.pdb' --pdb2_file='state2.pdb' --print_acc=1 --save_models=1 --print_detail=1 --atom_file='atom-important.xlsx' --res_file='res-important.xlsx'**

Important residues can be read in the res-important.xlsx file.

### Step 3. Network analysis
The community network analysis can be done by VMD. Sample **network.config** is provided here. Under the navigation of the important residues, one can avoid skewing the explanation of the network, together delivering unbias insights into the biased allosteric modulation mechanisms


### NOTE
Data used in the paper for the workflow are provided in figshare: Kong, Ming (2025). Data for Machine Learning Navigated Allosteric Network to Unveil Biased Allosteric Modulation of GPCRs. figshare. Dataset. https://doi.org/10.6084/m9.figshare.29262185
