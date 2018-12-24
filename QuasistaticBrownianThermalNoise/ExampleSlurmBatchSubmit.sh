#!/bin/bash -
#SBATCH -J TN_Cyl_Crystal             # Job Name
#SBATCH -o ElastostaticMirror.stdout  # Output file name
#SBATCH -e ElastostaticMirror.stderr  # Error file name
#SBATCH -n 240                        # Number of cores
#SBATCH --ntasks-per-node 20          # number of MPI ranks per node
#SBATCH -p orca-1                     # Queue name
#SBATCH -t 12:0:00                    # Run time

#Machine-specific module information...
#edit to set up your bash environment on your cluster
#this section is machine dependent
#you can comment out this section if you don't use modules

umask 0022 #others have read permission to the results
module purge
cp ~/bin/dealii_modules.sh .
source ./dealii_modules.sh
dealii_load_modules

module list #print loaded modules

# Run the code in a "Run/" subdirectory, copying
mkdir Run
cd Run
cp ../QuasistaticBrownianThermalNoise .
cp ../QuasistaticBrownianThermalNoise.cpp .
cp ../config.yaml .

mpirun -np 240 QuasistaticBrownianThermalNoise --configuration=./config.yaml &> QuasistaticBrownianThermalNoise.out
