# Installation

<br>  

## Cloning the Data Challenge

Clone the data challenge repo:  
```
git clone git@github.com:ocean-data-challenges/2024_DC_SSH_mapping_SWOT_OSE.git
```


## Creating the environment  

Create the data challenge conda environment, named env-dc-global-ose, by running the following command:
```
conda env create --file=dc_environment.yml 
```
and activate it with:

```
conda activate env-dc-global-ose
```

## Add the environment to jupyter

Add the environment to the available kernels for jupyter to see: 
```
ipython kernel install --name "env-dc-global-ose" --user
```
finally, select the "env-dc-global-ose" kernel in your notebook with Kernel > Change Kernel.
 


## You're now good to go ! 

