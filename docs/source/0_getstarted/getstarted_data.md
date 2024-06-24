# Download the data

<br> 

<br>  

The data are hosted and can be accessed on the MEOM server opendap [here](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/MEOM/OCEAN_DATA_CHALLENGES/2024_DC_SSH_mapping_SWOT_OSE/catalog.html). 

 

**A notebook to illustrate how to download and read the global data is available**

**If you are only interested in regional data, a notebook is available to read online the global data and download only regional data**

<br>

The dataset is presented with the following directory structure:

--- 

## 1) Data for experiment

**Nadir alongtrack data (L3 products) for SSH map reconstruction**

```
.
|-- alongtrack
``` 

--- 

## 2) Data for evaluation

**Independant nadir alongtrack data (L3 products) for SSH evaluation**

```
.
|-- independant_alongtrack
|   |-- al		% DT Altika Drifting Phase Global Ocean Along track SSALTO/DUACS Sea Surface Height L3 product
|   |   |-- 2023
|   |   |   |-- nrt_global_al_phy_l3_1hz_2023*.nc
|   |   |-- 2024
|   |   |   |-- nrt_global_al_phy_l3_1hz_2024*.nc
```

**Auxiliary data for diagnostics**

```
.
|-- sad
|   |-- distance_to_nearest_coastline_60.nc
|   |-- land_water_mask_60.nc
|   |-- variance_cmems_dt_allsat.nc

```

--- 

## 3) Data for comparison

**Reconstruction maps for comparison**

```
.
|-- maps
|   |-- mapping_miost_s3a_s3b_s6a-hr			% MIOST reconstruction 3 nadirs			
|   |-- mapping_miost_s3a_s3b_s6a-hr_swot		% MIOST reconstruction 3 nadirs + 1 SWOT
|   |-- mapping_miost_c2n_h2b_j3n_s3a_s3b_s6a-hr	% MIOST reconstruction 6 nadirs
|   |-- mapping_miost_c2n_h2b_j3n_s3a_s3b_s6a-hr	% MIOST reconstruction 6 nadirs + 1 SWOT
```

--- 


## Download the data

The data can be downloaded locally using the wget command. We recommand that the data be stored in the `data/` repository. 
For example, to download and unzip the experiment alongtrack data:

``` 
cd data/
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/OCEAN_DATA_CHALLENGES/2024_DC_SSH_mapping_SWOT_OSE/alongtrack/* 
tar -xvf alongtrack.tar.gz  
rm -f alongtrack.tar.gz
```


