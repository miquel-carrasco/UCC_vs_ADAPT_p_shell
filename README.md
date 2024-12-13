# UCC vs ADAPT for the nuclear shell model

This repository contains the scrips and data needed to perform the UCC and ADAPT VQE simulation of light 
nuclei in the *p* shell.

The **VQE** library has three modules: Nucleus, Ansatze and Methods. They have the necessary classes 
and functions to run a certain method (UCC or ADAPT), with a given *ansatz* representing an atomic nucleus.

The data necessary to bild the Hamiltonian matrix and the operators of each nucleus is stored in the **data** folder.
Such data was generated using the Cohen-Kourath interaction. Credit to Antonio Márquez Romero.

### VQE execution examples
In the **UCC_ADAPT_tests.py** script you will find two functions to execute both of the methods for any 
nuclei in the **data** folder. The name of the nuclei has to be given to the functions as a string (e. g. "Li6") 
and the reference state can be specified with an integer (corresponding to the many-body basis indices, 
specified in the **mb_basis_2.dat** file of the **data** folder).

