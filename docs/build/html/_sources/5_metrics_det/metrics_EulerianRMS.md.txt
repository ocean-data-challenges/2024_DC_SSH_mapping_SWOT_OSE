# Eulerian scores

<br>
 

<br> 

## Eulerian RMS


The Eulerian RMS (Root Mean Square) is a metric used to assess the difference between observed and reconstructed currents. It calculates the root of the sum of squared differences between the eastward and northward velocities of the drifter trajectory and the interpolated current at the same position.

Let's call $(u_d(p), v_d(p))$ the eastward and northward velocity derived from the drifter trajectory at the position p. The current to be compared is interpolated in time and space at position p and is called $(u(p), v(p))$.

$$RMS_u = \sqrt{\sum_p{(u(p) -u_d(p))^2}}$$
$$RMS_v = \sqrt{\sum_p{(v(p) -v_d(p))^2}}$$

A lower RMS value indicates better agreement between the observed and model currents.
 
<br>

 