# Products description

  
<br>

<hr>

## WOC products

- **BFN-QG**  

    For over 25 years, satellite altimetry has provided invaluable information about the ocean dynamics at many scales. Gridded Sea Surface Height (SSH) maps allow to estimate the mesoscale geostrophic circulation in the ocean. However, conventional interpolation techniques rely on static optimal interpolation schemes, hence limiting the estimation of non-linear dynamics at scales not well sampled by altimetry (i.e., below 150-200km at mid latitudes). To overcome this limitation in the resolution of small-scale SSH structures (and thus small-scale geostrophic currents), a Back-and-Forth Nudging algorithm combined with a Quasi-Geostrophic model, a technique called BFN-QG, has been implemented within the theme 1 of the ESA World Ocean Circulation (WOC) project. The performances have been evaluated within Observing System Experiments (OSEs) that use independent data (such as independent SSH, Sea Surface Temperature and drifter data) as ground-truth. By comparing the mapping performances to the ones obtained by operational products, the BFN-QG improves the mapping of short, energetic mesoscale structures and associated geostrophic currents both in space and time. In particular, the BFN-QG improves (i) the spatial effective resolution of the SSH maps by a factor of 20%, (ii) the zonal and (especially) the meridional geostrophic currents and (iii) the prediction of Lagrangian transport for lead times up to 10 days.

    **More information on the BFN-QG products [here](https://www.worldoceancirculation.org/Products#/metadata/7fe77c80-798a-42d4-a69c-2b5f0ba81a43) !**

<br>  

- **DDDcurr** 
 

    This dataset provides Level 4 total current including geostrophy and a data-driven approach for Ekman and near-inertial current, based on a convolution between drifter observation and wind history, to fit empirically a complex and time-lag dependant transfert function between ERA5 wind stress and current

    **More information on the Drifter data-driven products [here](https://www.worldoceancirculation.org/Products#/metadata/cfa65be8-b200-4243-aba0-843c363001d9) !**


<br>  

- **DADR-SR** 

    This dataset provides a level 4 (L4) gap free reconstruction of 2D ocean surface currents obtained through the combination of altimeter-derived geostrophic currents and satellite sea surface temperature (SST) L4 data. This WOC product uses the standard Altimeter-derived geostrophic currents (Copernicus operational product, ¼°) as background information and extracts additional dynamical information from the spatial and temporal gradients of a higher-resolution (HR) satellite-derived SST field (Copernicus OSTIA product, remapped onto a 1/10° grid). In particular, the patterns found in the HR satellite SST enable to compute correction factors that are assigned to the background geostrophic currents provided by the Altimeter system.

    **More information on the DADR-SR current products [here](https://www.worldoceancirculation.org/Products#/metadata/e84fe404-1ed3-4e6f-9b72-f75cd8eb7f7d) !**

<hr>

  
<br>

## GlobCurrent products

- **GC-combined** 

    The GlobCurrent data repository now includes the surface geostrophic current, the Ekman current at the surface and at 15 m depth, and the combined geostrophic and Ekman currents. The data are interpolated and collocated to a common grid with a spatial resolution of 25 km and a temporal resolution of 1 day for the geostrophic current and three hours for the Ekman currents and the combined currents. It covers the 23-year period from January 1993 to May 2016. A regional product for the Mediterranean Sea interpolated to a spatial resolution of 1/8 degree and a temporal resolution of 3 hours is also available.


    **More information on the GlobCurrent products [here](http://globcurrent.ifremer.fr/) !**

<hr>

<br>


## CMEMS products 

- **DUACS** 

    Altimeter satellite gridded Sea Level Anomalies (SLA) computed with respect to a twenty-year [1993, 2012] mean. The SLA is estimated by Optimal Interpolation, merging the L3 along-track measurement from the different altimeter missions available. Part of the processing is fitted to the Global Ocean. (see QUID document or http://duacs.cls.fr pages for processing details). The product gives additional variables (i.e. Absolute Dynamic Topography and geostrophic currents (absolute and anomalies)). It serves in near-real time applications. This product is processed by the DUACS multimission altimeter data processing system.


<hr>

<br>

## CLS products

- **MIOST** 
 

    Multi-scale interpolation (MIOST) combines ocean altimetry data along the track into continuous grids in time and space. Like the DUACS mapping system, it is based on a linear optimal interpolation scheme, with a different level of definition of the covariance functions. For this experimental product, only covariance functions representative of mesoscale geostrophic variability have been taken into account.


<hr>