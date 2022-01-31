# Physics-Informed Neural Networks for Fluid Dynamics :ocean:

## Authors
- Giulia Mescolini ([@giuliamesc](https://github.com/giuliamesc)) 
- Luca Sosta ([@sostaluca](https://gitlab.com/sostaluca))

## Content :books:
This repo contains the code for the course project "Numerical Analysis for Partial Differential Equations" at Politenico di Milano, MSc in Mathematical Engineering.

The project proposes an approach based on Physics-Informed Neural Networks to Fluid Dynamics problems.

## Code Structure :world_map:
- Folder `Examples`, containing the code for the solution with PINNs of all the test cases.
In each folder, you will find a folder containing the results of the simulation presented in the report, too.
    - `Poisson_Problem`: toy problem; solution of the Poisson equation on a square with mixed boundary conditions.
    - `Poiseuille_Flow`: laminar flow in a channel.
    - `Colliding_Flows`: colliding flows in a square.
    - `Cavity_Steady`: lid-driven cavity, steady version.
    - `Cavity_Unsteady`: lid-driven cavity, unsteady version.
    - `Coronary_Flow`: steady flow of blood in an arthery affected by a stenosis.
- Folder `Data Generation` contains the code for the generation of the numerical solution, exploited in the *Lid-Driven Cavity* and *Coronary Flow* test cases. The `coroParam` files contain the mesh for the *Coronary Flow* test case.
- Folder `Examples_Old` contains the deprecated version of the first test cases.
- Folder `Presentations` contains the presentations of this work to the tutors/to the classroom.
- `Report.pdf` is the project report, including all the results. 

## Instructions :dart:
1. Enter the folder of the test case of interest.
2. In the `.txt` file `simulation_options.txt` indicate the number of epochs, the amount of noise, the numerosity of the training data (divided by category: *collocation*, *fitting*, *boundary*) and test data.
3. Set `save_results = True` in the `.py` script if you wish to save the results and the loss trends in a folder automatically generated.
4. Run the script! :rocket:

