# Download the data

<br> 

<br>  

The data are hosted and can be accessed on the MEOM server opendap [here](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/MEOM/OCEAN_DATA_CHALLENGES/2024_DC_WOC-ESA/catalog.html). 

## Data information

The dataset is presented with the following directory structure. 

There is a repository for each region: Agulhas, Gulfstream and Mediterranean plus one repository for auxiliary data. In each region repository you'll find the same structure of zipped files: 

```
.
|-- DC_Agulhas 
|   |--  eval_Agulhas.tar.gz  
|   |--  obs_Agulhas.tar.gz 
|   |--  products_Agulhas.tar.gz  
.
|-- DC_Gulfstream 
|   |--  eval_Gulfstream.tar.gz  
|   |--  obs_Gulfstream.tar.gz 
|   |--  products_Gulfstream.tar.gz  
.
|-- DC_Mediterranean
|   |--  eval_Mediterranean.tar.gz  
|   |--  obs_Mediterranean.tar.gz 
|   |--  products_Mediterranean.tar.gz  
.
|-- Aux_data  
|   |-- sad.tar.gz  
```



### ``` obs_REGION.tar.gz```

The ```obs_REGION/``` repository contains the available observations in this region needed to map currents. The data can be stored in ```ssh/```, ```sst``` or ```drifters``` sub-repo and consists of netcdf files.  

### ``` eval_REGION.tar.gz```

The ```eval_REGION/``` repository contains the independant (i.e. not available in ```obs_REGION/```) observations that are used in the evaluation of the mapping performance.  

### ``` products_REGION.tar.gz```

The ```products_REGION/``` repository contains WOC products and other products that have already been evaluated on the data challenges, you can then check the performance of these products yourself and compare them to your methods. 
 


## Download and read the data

The data can be downloaded locally using the wget command. We recommand that the data be stored in the `data/` repository. 
For example, to download and unzip the observations needed to map currents in the Agulhas region use:


```
cd data/
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/OCEAN_DATA_CHALLENGES/2024_DC_WOC-ESA/DC_Agulhas/obs_Agulhas.tar.gz 
tar -xvf obs_Agulhas.tar.gz  
rm -f obs_Agulhas.tar.gz
```

**Example notebooks**


- A notebook to illustrate how to download and read the data is available: [download_and_acces_global_data.ipynb](../gallery/download_and_acces_global_data.ipynb)
 
