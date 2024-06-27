# Download the data

<br> 

<br>  

The data are hosted and can be accessed on the MEOM server opendap [here](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/MEOM/OCEAN_DATA_CHALLENGES/2024_DC_SSH_mapping_SWOT_OSE/catalog.html). 

 

**A notebook to illustrate how to download and read the global data is available**

**If you are only interested in regional data, a notebook is available to read online the global data and download only regional data**

<br>

The dataset is presented with the following directory structure:

--- 

## Data description

### 1) Data for experiment

**Nadir alongtrack data (L3 products) for SSH map reconstruction**
 
```
.
|-- nadirs
|   |-- c2n		% NRT Cryosat-2 new orbit Global Ocean Along track SSALTO/DUACS Sea Surface Height L3 product
|   |   |-- 2023
|   |   |   |-- nrt_global_c2n_phy_l3_1hz_2023*.nc
|   |   |-- 2024
|   |   |   |-- nrt_global_c2n_phy_l3_1hz_2024*.nc
|   |-- h2b		% NRT Haiyang-2B Global Ocean Along track SSALTO/DUACS Sea Surface Height L3 product
|   |   |-- 2023
|   |   |   |-- nrt_global_h2b_phy_l3_1hz_2023*.nc
|   |   |-- 2024
|   |   |   |-- nrt_global_h2b_phy_l3_1hz_2024*.nc
|   |-- j3n		% NRT Jason-3 Interleaved Global Ocean Along track SSALTO/DUACS Sea Surface Height L3 product
|   |   |-- 2023
|   |   |   |-- nrt_global_j3n_phy_l3_1hz_2023*.nc
|   |   |-- 2024
|   |   |   |-- nrt_global_j3n_phy_l3_1hz_2024*.nc
|   |-- s3a		% NRT Sentinel-3A Global Ocean Along track SSALTO/DUACS Sea Surface Height L3 product
|   |   |-- 2023
|   |   |   |-- nrt_global_s3a_phy_l3_1hz_2023*.nc
|   |   |-- 2024
|   |   |   |-- nrt_global_s3a_phy_l3_1hz_2024*.nc
|   |-- s3b		% NRT Sentinel-3B Global Ocean Along track SSALTO/DUACS Sea Surface Height L3 product
|   |   |-- 2023
|   |   |   |-- nrt_global_s3b_phy_l3_1hz_2023*.nc
|   |   |-- 2024
|   |   |   |-- nrt_global_s3b_phy_l3_1hz_2024*.nc
|   |-- s6a_hr		% NRT Sentinel-6A (SAR mode) Global Ocean Along track SSALTO/DUACS Sea Surface Height L3 product
|   |   |-- 2023
|   |   |   |-- nrt_global_s6a_hr_phy_l3_1hz_2023*.nc
|   |   |-- 2024
|   |   |   |-- nrt_global_s6a_hr_phy_l3_1hz_2024*.nc
|   |-- swon		%  NRT SWOT nadir 28days Global Ocean Along track SSALTO/DUACS Sea Surface Height L3 product
|   |   |-- 2023
|   |   |   |-- nrt_global_swo_hr_phy_l3_1hz_2023*.nc
|   |   |-- 2024
|   |   |   |-- nrt_global_swo_hr_phy_l3_1hz_2024*.nc
``` 


**SWOT Karin data (L3 products) for SSH map reconstruction**

```
.
|-- karin
``` 

--- 

### 2) Data for evaluation

**Independant nadir alongtrack data (L3 products) for SSH evaluation**

```
.
|-- indep_nadir
|   |-- al		% NRT Altika Drifting Phase Global Ocean Along track SSALTO/DUACS Sea Surface Height L3 product
|   |   |-- 2023
|   |   |   |-- nrt_global_al_phy_l3_1hz_2023*.nc
|   |   |-- 2024
|   |   |   |-- nrt_global_al_phy_l3_1hz_2024*.nc

```


**Independant drifter data (L3 products) for geostrophic current evaluation**

```
.
|-- indep_drifters 

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

### 3) Data for comparison

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


## Downloading the data

The data can be downloaded locally directly from your browser by clicking here: [Download](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/MEOM/OCEAN_DATA_CHALLENGES/2024_DC_SSH_mapping_SWOT_OSE/catalog.html)

</br>

or by using the wget command, for example, to download and unzip the experiment nadir data: 

``` 
cd data/
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/OCEAN_DATA_CHALLENGES/2024_DC_SSH_mapping_SWOT_OSE/nadirs.tar.gz 
tar -xvf nadirs.tar.gz  
rm -f nadirs.tar.gz
```

</br>


Either way, we recommand that the data be then stored in the `data/` repository.  

