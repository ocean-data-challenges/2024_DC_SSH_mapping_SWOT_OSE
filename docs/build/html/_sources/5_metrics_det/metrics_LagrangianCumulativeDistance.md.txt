# Lagrangian diagnostics

<br>
 

<br>

## Lagrangian Cumulative Distance
  


The Lagrangian Cumulative Distance measures the cumulative separation between real drifters and synthetic drifters over time. For each drifter and each day, an ensemble of particles around the drifter position is released and advected with the velocity field to be validated.

For each drifter, for each day, an ensemble of particles around the drifter position are seeded (1 particle every km in a 12.5km radius). These particles are then advected with the velocity field to be validated and compared with the real drifter trajectory. 

For each particle $p$, we compute the normalized cumulative separation distance between the real drifter and the synthetic drifter from each product $s_p$ as defined in Liu and Weisberg (2011):

$$s_p[T] = \frac{\sum_{i=t0}^{T} d_p [t]}{\sum_{i=t0}^{T} length[t]} $$

with $d_p[t]$ the Euclidian distance between the virtual particle $p$ and the in situ drifter positions and $length$ the length of the drifter trajectory after a time t of advection from the drifter initial position at time $t0$. $s_p$ are the scores computed for every day T within the period of advection. 

This metric quantifies how well the reconstructed currents can reproduce the drifter's trajectory over time, providing insights into long-term accuracy.

 
<br>

## References 

- Liu, Y., & Weisberg, R. H. (2011). Evaluation of trajectory modeling in different dynamic regions using normalized cumulative Lagrangian separation. Journal of Geophysical Research: Oceans, 116(C9).
 
 
 