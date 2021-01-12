This project contains the code used for the experimental results of the paper
"Efficient Exact Computation of Setwise Minimax Regret forInteractive Preference Elicitation" (AAMAS 2021).
Authors: Federico Toffano, Paolo Viappiani, and Nic Wilson.

The file TestRandom.py runs experiments with the settings used for of Table 1, Table 2, Table 3, Table 4 and Figure 3.
The datasets are randomly generated at each execution

The file TestRandom.txt runs experiments with the settings used for of Table 5. The datasets used are stored in the
folder "dataset" and briefly described in the paper.

Software requirements:
Cplex (our version is 12.9) https://www.ibm.com/support/pages/downloading-ibm-ilog-cplex-optimization-studio-v1290
Pycddlib (our version is 2.1) https://pypi.org/project/pycddlib/#history
Python-sat (our version is 0.1.4.dev9) https://github.com/pysathq/pysat

