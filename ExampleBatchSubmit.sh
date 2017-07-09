#!/bin/bash -
#PBS -l nodes=24:ppn=12
#PBS -l walltime=12:0:00


#PBS -N TN_Cyl_Crystal
#PBS -o ElastostaticMirror.stdout
#PBS -e ElastostaticMirror.stderr
#PBS -d .
#PBS -W umask=022
#PBS -S /bin/bash

#Machine-specific module information...
#edit to set up your bash environment on your cluster
#this section is machine dependent
#you can comment out this section if you don't use modules

umask 0022 #others have read permission to the results
module purge
export MODULEPATH=/share/apps/Modules/modulefiles:$MODULEPATH
set -x
export PATH=$(pwd -P)/bin:$PATH
cp ~/bin/LoadDealiiModules_v2.bash .
. LoadDealiiModules_v2.bash #example script to load any necessary modules
                            #e.g. "module load dealii"
module list #print the loaded modules

# Run the code in a "Run/" subdirectory, copying
mkdir Run
cd Run
cp ../QuasistaticBrownianThermalNoise .
cp ../QuasistaticBrownianThermalNoise.cpp .

mpirun -np 288 QuasistaticBrownianThermalNoise -ksp_monitor -snes_monitor -options_table &> QuasistaticBrownianThermalNoise.out



