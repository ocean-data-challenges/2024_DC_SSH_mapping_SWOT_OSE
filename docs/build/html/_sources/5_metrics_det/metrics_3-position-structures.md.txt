# Position of structures

<br>
 

<br>

## Current direction using Fronts

The first step of the algorithm is to compute the flow across the fronts for each velocity product to be validated.
The fronts used in this analysis are the one computed within the WOC project from the Microwave and SEVIRI SST observation. The fronts are smoothed using a spline filter before the analyses. 



Let’s call , the zonal and meridional component of the velocity product k, and lon, lat the coordinates that correspond to the front position.
For each point P that belongs to a front, for each velocity product k, we compute the flow across the velocity:

with 

The smaller the flow, the more accurate is the velocity as we assum e here that the velocity is transporting the tracer.

## Eddy position and radius using Eddies detected using SST

 
 
## References 