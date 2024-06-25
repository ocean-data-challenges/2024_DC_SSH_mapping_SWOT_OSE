.. 2024_DC_SSH_mapping_SWOT_OSE documentation master file, created by
   sphinx-quickstart on Fri Jul 21 14:53:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/dc_2024_ose_global_banner.jpg
    :width: 1500
    :alt: alternate text
    :align: center 
     
     
.. raw:: html
 
    <embed>  
        </br>
        
        </hr>
        
        </br>
        
    </embed>



Real-time observation of ocean surface topography is essential for various oceanographic applications. Historically, these observations relied mainly on satellite nadir altimetry data, which were limited to observe scales greater than approximately 60 km. However, the recent launch of the wide-swath SWOT mission in December 2022 marks a significant advancement, enabling the two-dimensional observation of finer oceanic scales (~15 km). While the direct analysis of the two-dimensional content of these swaths can provide valuable insights into ocean surface dynamics, integrating such data into mapping systems presents several challenges. This data-challenge focuses on integrating the SWOT mission into multi-mission mapping systems. Specifically, it examines the contribution of the SWOT mission to both the current nadir altimetry constellation (seven nadirs) and a reduced nadir altimetry constellation (three nadirs). 

Several mapping techniques, such as statistical interpolation methods or ocean model assimilation methods, are currently proposed to provide operational maps of ocean surface heights and currents. 
New mapping techniques (e.g. data-driven methods) are emerging and being tested in a research and development context. 
It is therefore becoming important to inform users and developers about the accuracy of scale represented by each mapping system. A sensitivity study of different mapping methods in a SWOT context is also proposed.

     
.. raw:: html


    <embed> 
        <p align="center">
          <img src="_static/dc_2024_ose_karin.png" alt="Alt Text" width="800"/>
        </p>

        <p align="center">
            <i>
            Figure: Example of geostrophic current reconstruction on 2023-08-31 with MIOST at a) global scale, b) view from Karin L3 products over the Agulhas region, c) from MIOST reconstruction integration 1SWOT and 6 nadirs, d) from MIOST reconstruction integration 6 nadirs and e) the difference in MIOST reconstructions between integration 1SWOT and 6 nadirs vs 6 nadirs only
            </i>
        </p>

    </embed>
    
The goal of the present data-challenge is:
1) to investigate the contribution of SWOT KaRin data in global & regional mapping systems
2) to investigate how to best reconstruct sequences of Sea Surface Height (SSH) from partial nadir and **KaRin** satellite altimetry observations and in using various mapping method (dynamical, data-driven approach...)

This data challenge follows an Observation System Experiment framework: Satellite observations are from real sea surface height data from altimeter. 
The practical goal of the challenge is to investigate the contribution of SWOT KaRin data and the best mapping method according to scores described below and in Jupyter notebooks.
 
    
     

     
.. raw:: html
 
    <embed>  
        </br>
        
        </hr>
        
        </br>
        
    </embed>
    

Observations  
============


Nadirs sea-level anomaly Level 3 products
-----------------------------------------

To produce the gridded sea level maps, we used the global ocean sea level anomaly observations from the Near-Real-Time (NRT) Level-3 altimeter satellite along-track data distributed by the EU Copernicus Marine Service (product reference [SEALEVEL_GLO_PHY_L3_NRT_008_044](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L3_NRT_008_044/description)), specifically for the Jason-3, Sentinel-3A, Sentinel-3B, Sentinel-6A, SARAL-Altika, Cryosat-2, Haiyang-2B, missions. This dataset covers the global ocean and is available at a sampling rate of 1 Hz (approximately 7 km spatial spacing).

SWOT sea-level anomaly Level 3 products
---------------------------------------

In addition to the nadir altimetry constellation previously mentioned, we conducted experiments involving the integration of SWOT Level-3 Ocean product (specifically referencing [SWOT_L3_SSH](https://www.aviso.altimetry.fr/en/data/products/sea-surface-height-products/global/swot-l3-ocean-products.html)) during the 21-day phase of the mission. The SWOT_L3_SSH product combines ocean topography measurements collected from both the SWOT KaRIn and nadir altimeter instruments, consolidating them into a unified variable on a 2 km spatial grid spacing. For our investigation, we used version 0.3 & version 1.0 of the product accessible through the AVISO+ portal (AVISO/DUACS, 2023). These data were derived from the Level-2 "Beta Pre-validated" KaRIn Low Rate (Ocean) product (NASA/JPL and CNES).


     
.. raw:: html
 
    <embed> 
        <p align="center">
          <img src="_static/sampling.png" alt="Alt Text" width="1000"/>
        </p>
        <p align="center">
        <i>
        Figure: Example of spatial sampling in the North Atlantic of a) 3 nadirs altimeters, b) 7 nadirs altimeters and c) 1 SWOT
        </i>
        </p>
        </br>
        
        </hr>
        
        </br>
    </embed>


Data sequence and use  
=====================
 
The SSH reconstructions are assessed at global scale and over the period from 2023-08-01 to 2024-05-01.
The SSH reconstructions can also be assessed at regional scale and over the same period from 2023-08-01 to 2024-05-01.

For reconstruction methods that need a spin-up, the **observations** from other period can be used.

The altimeter data from Saral/AltiKa data mentioned above should never be used so that any reconstruction can be considered uncorrelated to the evaluation period.
     
     
.. raw:: html
 
    <embed>  
        </br>
        
        </hr>
        
        </br>
        
    </embed>
    
     
More data challenges   
====================

If you are interested in more data challenges relating to oceanographic data (global altimetric mapping, SWOT preprocessing techniques ...), you can visit the ocean-data-challenges website. 
  
  
    
.. raw:: html  


    <embed>  
        
        <br />
        
        <center><a  href="https://ocean-data-challenges.github.io"/> <img src="_static/odc_webpage.jpg" alt="Alt Text" width="500"/></a></center>
        
        <center><a  href="https://ocean-data-challenges.github.io" alt="Alt Text"/> ocean-data-challenges.github.io </a></center>
        
        <br /> 
        
        <br />
        
          
        <br />
        
        <hr />
        
        <br />
        
        <center> <img src="_static/logos_partenaires_DC_mapping_SSH_SWOT.jpg" alt="Alt Text" width="1000"/></center>
        
        
        
        <br />
        
        <hr />
        
        <br />
        
         
        So far, the github page visits amount to: <br> <br> <a href="https://github.com/ocean-data-challenges/2024_DC_WOC-ESA"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Focean-data-challenges%2F2024_DC_WOC-ESA&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=PAGE+VIEWS&edge_flat=false"/></a> 
        
    </embed>
    
-----------------  

   
    
.. raw:: html

    <embed>  
    
        
        <br /> 
        
    </embed>
 
    
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Participate 

   0_getstarted/index.md
   
    
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Mapping methods 

   1_products/index.md 

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Global Evaluation
 
   2_globaleval/index.md 

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Regional Evaluations
 
   3_regionaleval/index.md 
      
   
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Metrics details 
 
   5_metrics_det/index.md  
    

   

.. toctree::  
    :caption: Contact us
    :hidden:
    :maxdepth: 0 
    
    contactus.md  
   
    
