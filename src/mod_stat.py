import logging

import matplotlib.pylab as plt
import numpy as np
import pyinterp
import xarray as xr
from netCDF4 import Dataset
from scipy import stats

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

from src.mod_filter import *
from src.mod_interp import *


def bin_data(ds, output_file, lon_out=np.arange(0, 360, 1), lat_out=np.arange(-90, 90, 1), freq_out='1D'):
    
    binning = pyinterp.Binning2D(pyinterp.Axis(lon_out, is_circle=True),
                                 pyinterp.Axis(lat_out))
    
    
    # All spatial scale     
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['sla_unfiltered']).compute()
    mean_track = binning.variable('mean')
    variance_track = binning.variable('variance')
    nb_pts_track = binning.variable('count')
    time_mean_track = (ds['sla_unfiltered'].resample(time=freq_out)).mean()
    time_variance_track = (ds['sla_unfiltered'].resample(time=freq_out)).var()
    time_rms_track = np.sqrt(((ds['sla_unfiltered']**2).resample(time=freq_out)).mean())
    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['msla_interpolated']).compute()
    mean_msla = binning.variable('mean')
    variance_msla = binning.variable('variance')
    nb_pts_msla = binning.variable('count')
    time_mean_msla = (ds['msla_interpolated'].resample(time=freq_out)).mean()
    time_variance_msla = (ds['msla_interpolated'].resample(time=freq_out)).var()
    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['mapping_err']).compute()
    mean_mapping_err = binning.variable('mean')
    variance_mapping_err = binning.variable('variance')
    nb_pts_mapping_err = binning.variable('count')
    time_mean_mapping_err = (ds['mapping_err'].resample(time=freq_out)).mean()
    time_variance_mapping_err = (ds['mapping_err'].resample(time=freq_out)).var()
    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['mapping_err']**2).compute()
    rmse = np.sqrt(binning.variable('mean'))
    time_mean_rmse = np.sqrt(((ds['mapping_err']**2).resample(time=freq_out)).mean())
    
    
    ds1 = xr.Dataset({'mean_track' : (('lat', 'lon'), mean_track.T), 
                      'variance_track' : (('lat', 'lon'), variance_track.T),
                      'mean_msla' : (('lat', 'lon'), mean_msla.T), 
                      'variance_msla' : (('lat', 'lon'), variance_msla.T),
                      'mean_mapping_err' : (('lat', 'lon'), mean_mapping_err.T), 
                      'variance_mapping_err' : (('lat', 'lon'), variance_mapping_err.T),
                      'rmse' : (('lat', 'lon'), rmse.T),
                      'count' : (('lat', 'lon'), nb_pts_mapping_err.T),
                      
                      'timeserie_mean_track' : (('time'), time_mean_track.data), 
                      'timeserie_variance_track' : (('time'), time_variance_track.data),
                      'timeserie_rms_track' : (('time'), time_rms_track.data),
                      'timeserie_mean_msla' : (('time'), time_mean_msla.data), 
                      'timeserie_variance_msla' : (('time'), time_variance_msla.data),
                      'timeserie_mean_mapping_err' : (('time'), time_mean_mapping_err.data), 
                      'timeserie_variance_mapping_err' : (('time'), time_variance_mapping_err.data),
                      'timeserie_rmse' : (('time'), time_mean_rmse.data),
                      
                      },
                      coords={'lon': lon_out, 
                              'lat': lat_out,
                              'time': time_mean_rmse['time'],
                               }
                       )
    
    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['sla_filtered']).compute()
    mean_track = binning.variable('mean')
    variance_track = binning.variable('variance')
    time_variance_track = (ds['sla_filtered'].resample(time=freq_out)).var()
    time_rms_track = np.sqrt(((ds['sla_filtered']**2).resample(time=freq_out)).mean())
    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['msla_filtered']).compute()
    mean_msla = binning.variable('mean')
    variance_msla = binning.variable('variance')
    time_mean_msla = (ds['msla_filtered'].resample(time=freq_out)).mean()
    time_variance_msla = (ds['msla_filtered'].resample(time=freq_out)).var()
    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['mapping_err_filtered']).compute()
    mean_mapping_err = binning.variable('mean')
    variance_mapping_err = binning.variable('variance')
    nb_pts_mapping_err = binning.variable('count')
    time_mean_mapping_err = (ds['mapping_err_filtered'].resample(time=freq_out)).mean()
    time_variance_mapping_err = (ds['mapping_err_filtered'].resample(time=freq_out)).var()
    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['mapping_err_filtered']**2).compute()
    rmse = np.sqrt(binning.variable('mean'))
    time_mean_rmse = np.sqrt(((ds['mapping_err_filtered']**2).resample(time=freq_out)).mean())
    
    ds2 = xr.Dataset({'mean_track' : (('lat', 'lon'), mean_track.T), 
                      'variance_track' : (('lat', 'lon'), variance_track.T),
                      'mean_msla' : (('lat', 'lon'), mean_msla.T), 
                      'variance_msla' : (('lat', 'lon'), variance_msla.T),
                      'mean_mapping_err' : (('lat', 'lon'), mean_mapping_err.T), 
                      'variance_mapping_err' : (('lat', 'lon'), variance_mapping_err.T),
                      'rmse' : (('lat', 'lon'), rmse.T),
                      'count' : (('lat', 'lon'), nb_pts_mapping_err.T),
                      
                      'timeserie_mean_track' : (('time'), time_mean_track.data), 
                      'timeserie_variance_track' : (('time'), time_variance_track.data),
                      'timeserie_rms_track' : (('time'), time_rms_track.data),
                      'timeserie_mean_msla' : (('time'), time_mean_msla.data), 
                      'timeserie_variance_msla' : (('time'), time_variance_msla.data),
                      'timeserie_mean_mapping_err' : (('time'), time_mean_mapping_err.data), 
                      'timeserie_variance_mapping_err' : (('time'), time_variance_mapping_err.data),
                      'timeserie_rmse' : (('time'), time_mean_rmse.data),
                      },
                      coords={'lon': lon_out, 
                              'lat': lat_out, 
                              'time': time_mean_rmse['time']
                               }
                       )


    ds1.to_netcdf(output_file, group="all_scale", format="NETCDF4")
    ds2.to_netcdf(output_file, "a", group="filtered", format="NETCDF4")


    
    
def compute_statistical_test(ds_ref, ds_study, variable, grid_lon=np.arange(0., 360., 1.0), grid_lat=np.arange(-90., 90., 1.0)):
    # ds_ref = ds_interp_3nadirs['mapping_err']
    # ds_interp_6nadirs['mapping_err']
    from scipy.stats import ttest_rel

    delta_lat = 1
    delta_lon = 1
    
    lat = ds_ref.latitude.values
    lon = ds_ref.longitude.values


    data_t_statistic = np.zeros((grid_lat.size, grid_lon.size))
    data_p_value = np.zeros((grid_lat.size, grid_lon.size))
    rmse1 = np.zeros((grid_lat.size, grid_lon.size))
    rmse2 = np.zeros((grid_lat.size, grid_lon.size))

    jj = 0
    for ilat in grid_lat:
        # print(ilat)

        lat_min = ilat - 0.5*delta_lat
        lat_max = ilat + 0.5*delta_lat

        selected_lat_index = np.where(np.logical_and(lat >= lat_min, lat <= lat_max))[0]
        err_ref_tmp = ds_ref[variable].values[selected_lat_index]
        err_study_tmp = ds_study[variable].values[selected_lat_index]

        ii = 0

        for ilon in grid_lon % 360.:

            lon_min = ilon - 0.5*delta_lon
            lon_max = ilon + 0.5*delta_lon

            if (lon_min < 0.) and (lon_max > 0.):
                selected_index = np.where(np.logical_or(lon[selected_lat_index] % 360. >= lon_min + 360.,
                                                              lon[selected_lat_index] % 360. <= lon_max))[0]
            elif (lon_min > 0.) and (lon_max > 360.):
                selected_index = np.where(np.logical_or(lon[selected_lat_index] % 360. >= lon_min,
                                                              lon[selected_lat_index] % 360. <= lon_max - 360.))[0]
            else:
                selected_index = np.where(np.logical_and(lon[selected_lat_index] % 360. >= lon_min,
                                                               lon[selected_lat_index] % 360. <= lon_max))[0]

                if len(selected_index) > 0:

                    selected_data_ref = np.ma.masked_where(err_ref_tmp[selected_index].flatten() > 1.E10,
                                                               err_ref_tmp[selected_index].flatten())
                    selected_data_study = np.ma.masked_where(err_study_tmp[selected_index].flatten() > 1.E10,
                                                               err_study_tmp[selected_index].flatten())

                    t_statistic, p_value = ttest_rel(selected_data_ref**2, selected_data_study**2)

                    data_t_statistic[jj, ii] = t_statistic
                    data_p_value[jj, ii] = p_value
                    rmse1[jj, ii] = np.sqrt(np.mean(selected_data_ref**2))
                    rmse2[jj, ii] = np.sqrt(np.mean(selected_data_study**2))

            ii += 1

        jj += 1
        
    
    return rmse1, rmse2, data_t_statistic, data_p_value



    

def compute_stat_scores(ds_interp, lambda_min, lambda_max, output_file):
    
    #logging.info("Interpolate SLA maps onto alongtrack")
    #ds_interp = run_interpolation(ds_maps, ds_alongtrack)
     
    logging.info("Compute mapping error all scales")
    ds_interp['mapping_err'] = ds_interp['msla_interpolated'] - (ds_interp['sla_unfiltered'] - ds_interp['lwe'])
    
    logging.info("Compute mapping error for scales between %s and %s km", str(lambda_min), str(lambda_max))
    # Apply bandpass filter
    ds_interp = apply_bandpass_filter(ds_interp, lambda_min=lambda_min, lambda_max=lambda_max)
    
    logging.info("Compute binning statistics")
    # Bin data maps
    bin_data(ds_interp, output_file)
    
    logging.info("Compute statistics by oceanic regime")
    compute_stat_scores_by_regimes(ds_interp, output_file)
    
    logging.info("Stat file saved as: %s", output_file)

    
def dm_test(e1, e2, h = 1):
    
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Length of lists (as real numbers)
    T = e1.size
        
    d_lst = (e1 - e2)
    # Mean of d        
    mean_d = np.mean(d_lst)
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    
    
    return DM_stat, p_value


    
def compute_stat_scores_intercompare(ds_interp1, ds_interp2, lambda_min, lambda_max, output_file):
    
    #logging.info("Interpolate SLA maps onto alongtrack")
    #ds_interp = run_interpolation(ds_maps, ds_alongtrack)
     
    logging.info("Compute mapping error all scales")
    ds_interp1['mapping_err'] = ds_interp1['msla_interpolated'] - (ds_interp1['sla_unfiltered'] - ds_interp1['lwe'])
    
    ds_interp2['mapping_err'] = ds_interp2['msla_interpolated'] - (ds_interp2['sla_unfiltered'] - ds_interp1['lwe'])
    
    logging.info("Compute mapping error for scales between %s and %s km", str(lambda_min), str(lambda_max))
    # Apply bandpass filter
    ds_interp1 = apply_bandpass_filter(ds_interp1, lambda_min=lambda_min, lambda_max=lambda_max)
    ds_interp2 = apply_bandpass_filter(ds_interp2, lambda_min=lambda_min, lambda_max=lambda_max)
    
    lon_out=np.arange(0, 360, 1)
    lat_out=np.arange(-90, 90, 1)
    
    logging.info("Compute binning statistics")
    # Bin data maps
    
    rmse_ref, rmse_study, data_t_statistic, data_p_value = compute_statistical_test(ds_interp1, ds_interp2, variable='mapping_err')
    
    ds1 = xr.Dataset({
                      'rmse_ref' : (('lat', 'lon'), rmse_ref),
                      'rmse_study' : (('lat', 'lon'), rmse_study),
                      't_statistic' : (('lat', 'lon'), data_t_statistic), 
                      'p_value' : (('lat', 'lon'), data_p_value), 
                      },
                      coords={'lon': lon_out, 
                              'lat': lat_out,
                               }
                       )
    
    ds1.to_netcdf(output_file, group="all_scale", format="NETCDF4")
    
    rmse_ref2, rmse_study2, data_t_statistic2, data_p_value2 = compute_statistical_test(ds_interp1, ds_interp2, variable='mapping_err_filtered')
    
    ds2 = xr.Dataset({
                      'rmse_ref' : (('lat', 'lon'), rmse_ref2),
                      'rmse_study' : (('lat', 'lon'), rmse_study2),
                      't_statistic' : (('lat', 'lon'), data_t_statistic2), 
                      'p_value' : (('lat', 'lon'), data_p_value2), 
                      },
                      coords={'lon': lon_out, 
                              'lat': lat_out,
                               }
                       )
    
    
    ds2.to_netcdf(output_file, "a", group="filtered", format="NETCDF4")
    
    logging.info("Stat file saved as: %s", output_file)
    
       
def compute_stat_scores_by_regimes(ds_interp, output_file): 
    
    # distance_to_nearest_coast = '/home/ad/ballarm/data/wisa/data_challenge_ose//data/sad/distance_to_nearest_coastline_60.nc'
    # land_sea_mask = '/home/ad/ballarm/data/wisa/data_challenge_ose//data/sad/land_water_mask_60.nc'
    # variance_ssh = '/home/ad/ballarm/data/wisa/data_challenge_ose//data/sad/variance_cmems_dt_allsat.nc'
    distance_to_nearest_coast = '../data/sad/distance_to_nearest_coastline_60.nc'
    land_sea_mask = '../data/sad/land_water_mask_60.nc'
    variance_ssh = '../data/sad/variance_cmems_dt_allsat.nc'
    variance_criteria = 0.02             # min variance contour in m**2 to define the high variability regions
    coastal_distance_criteria = 200.     # max distance to coast in km to define the coastal regions
    
    lon_vector = ds_interp['longitude'].values
    lat_vector = ds_interp['latitude'].values
        
    # interpolate distance_to_nearest_coast, land_sea_mask, variance_ssh to lon/lat alongtrack
    ds = xr.open_dataset(land_sea_mask)
    x_axis = pyinterp.Axis(ds['lon'][:], is_circle=True)
    y_axis = pyinterp.Axis(ds['lat'][:])
    lsm = ds.variables["mask"][:].T
    
    lsm = np.where(lsm > 1, 1.0, lsm)
    # The undefined values must be set to nan.
    # lsm[lsm.mask] = float("nan")
    grid = pyinterp.Grid2D(x_axis, y_axis, lsm)
    lsm_interp = pyinterp.bivariate(grid, lon_vector, lat_vector, interpolator='nearest').reshape(lon_vector.shape)
    #plt.scatter(lon_vector[:10000], lat_vector[:10000], c=lsm_interp[:10000], s=10)
    #plt.show()
    
    ds = xr.open_dataset(distance_to_nearest_coast)
    x_axis = pyinterp.Axis(ds['lon'][:], is_circle=True)
    y_axis = pyinterp.Axis(ds['lat'][:])
    distance = ds.variables["distance"][:].T
    # The undefined values must be set to nan.
    # distance[distance.mask] = float("nan")
    grid = pyinterp.Grid2D(x_axis, y_axis, distance.data)
    distance_interp = pyinterp.bivariate(grid, lon_vector, lat_vector).reshape(lon_vector.shape)
    
    ds = xr.open_dataset(variance_ssh)
    x_axis = pyinterp.Axis(ds['longitude'][:], is_circle=True)
    y_axis = pyinterp.Axis(ds['latitude'][:])
    variance = ds.variables["sla"][:].T
    # The undefined values must be set to nan.
    # distance[distance.mask] = float("nan")
    grid = pyinterp.Grid2D(x_axis, y_axis, variance.data)
    variance_interp = pyinterp.bivariate(grid, lon_vector, lat_vector).reshape(lon_vector.shape)
    
    # Clean with lsm
    msk_land_data = np.ma.masked_where(lsm_interp == 1, lsm_interp).mask
    msk_coastal_data = np.ma.masked_where(distance_interp <= coastal_distance_criteria, distance_interp).mask
    msk_offshore_data = np.ma.masked_where(distance_interp >= coastal_distance_criteria, distance_interp).mask
    msk_lowvar_data = np.ma.masked_where(variance_interp <= variance_criteria, variance_interp).mask
    msk_highvar_data = np.ma.masked_where(variance_interp >= variance_criteria, variance_interp).mask
    msk_extra_equatorial_band = np.ma.masked_where(np.abs(lat_vector) > 10, lat_vector).mask
    msk_equatorial_band = np.ma.masked_where(np.abs(lat_vector) < 10, lat_vector).mask
    msk_below_arctic = np.ma.masked_where(lat_vector < 70., lat_vector).mask
    msk_above_antarctic = np.ma.masked_where(lat_vector > -70., lat_vector).mask
    msk_arctic = np.ma.masked_where(lat_vector > 70., lat_vector).mask
    msk_antarctic = np.ma.masked_where(lat_vector < -70., lat_vector).mask
    
    for var_name in ['mapping_err', 'msla_interpolated', 'sla_unfiltered', 'mapping_err_filtered', 'msla_filtered', 'sla_filtered']:
        data_vector = ds_interp[var_name].values
    
        # distance <= 200km
        msk = msk_land_data + msk_offshore_data
        data_vector_selected = np.ma.masked_where(msk, data_vector).compressed()
        if data_vector_selected.size > 0:
            coastal_analysis = stats.describe(data_vector_selected, nan_policy='omit')
            coastal_rmse = np.sqrt(np.nanmean((np.ma.masked_invalid(data_vector_selected))**2))
            #lon_vector_selected = np.ma.masked_where(msk, lon_vector).compressed()
            #lat_vector_selected = np.ma.masked_where(msk, lat_vector).compressed()
            #plt.scatter(lon_vector_selected, lat_vector_selected, c=data_vector_selected, s=10)
            #plt.show()
        else:
            coastal_analysis = [0, [np.nan, np.nan], np.nan, np.nan, np.nan, np.nan,]
            coastal_rmse = np.nan
    
        # distance >= 200km & variance >= 0.02 & extra equatorial
        msk = msk_land_data + msk_coastal_data + msk_lowvar_data + msk_equatorial_band + msk_arctic + msk_antarctic
        data_vector_selected = np.ma.masked_where(msk, data_vector).compressed()
        if data_vector_selected.size > 0:
            offshore_highvar_analysis = stats.describe(data_vector_selected, nan_policy='omit')
            offshore_highvar_rmse = np.sqrt(np.nanmean((np.ma.masked_invalid(data_vector_selected))**2))
            #lon_vector_selected = np.ma.masked_where(msk, lon_vector).compressed()
            #lat_vector_selected = np.ma.masked_where(msk, lat_vector).compressed()
            #plt.scatter(lon_vector_selected, lat_vector_selected, c=data_vector_selected, s=10)
            #plt.show()
        else:
            offshore_highvar_analysis = [0, [np.nan, np.nan], np.nan, np.nan, np.nan, np.nan,]
            offshore_highvar_rmse = np.nan
    
        # distance >= 200km & variance <= 0.02 & extra equatorial
        msk = msk_land_data + msk_coastal_data + msk_highvar_data + msk_equatorial_band + msk_arctic + msk_antarctic
        data_vector_selected = np.ma.masked_where(msk, data_vector).compressed()
        if data_vector_selected.size > 0:
            offshore_lowvar_analysis = stats.describe(data_vector_selected, nan_policy='omit')
            offshore_lowvar_rmse = np.sqrt(np.nanmean((np.ma.masked_invalid(data_vector_selected))**2))
            #lon_vector_selected = np.ma.masked_where(msk, lon_vector).compressed()
            #lat_vector_selected = np.ma.masked_where(msk, lat_vector).compressed()
            #plt.scatter(lon_vector_selected, lat_vector_selected, c=data_vector_selected, s=10)
            #plt.show()
        else:
            offshore_lowvar_analysis = [0, [np.nan, np.nan], np.nan, np.nan, np.nan, np.nan,]
            offshore_lowvar_rmse = np.nan
    
        # Equatorial band & distance >= 200km
        msk = msk_land_data + msk_extra_equatorial_band + msk_coastal_data  + msk_arctic + msk_antarctic
        data_vector_selected = np.ma.masked_where(msk, data_vector).compressed()
        if data_vector_selected.size > 0:
            equatorial_analysis = stats.describe(data_vector_selected, nan_policy='omit')
            equatorial_rmse = np.sqrt(np.nanmean((np.ma.masked_invalid(data_vector_selected))**2))
        else:
            equatorial_analysis = [0, [np.nan, np.nan], np.nan, np.nan, np.nan, np.nan,]
            equatorial_rmse = np.nan
    
        # Arctic
        msk = msk_land_data + msk_below_arctic
        data_vector_selected = np.ma.masked_where(msk, data_vector).compressed()
        if data_vector_selected.size > 0:
            arctic_analysis = stats.describe(data_vector_selected, nan_policy='omit')
            arctic_rmse = np.sqrt(np.nanmean((np.ma.masked_invalid(data_vector_selected))**2))
        else:
            arctic_analysis = [0, [np.nan, np.nan], np.nan, np.nan, np.nan, np.nan,]
            arctic_rmse = np.nan
        
        # AntArctic
        msk = msk_land_data + msk_above_antarctic
        data_vector_selected = np.ma.masked_where(msk, data_vector).compressed()
        if data_vector_selected.size > 0:
            antarctic_analysis = stats.describe(data_vector_selected, nan_policy='omit')
            antarctic_rmse = np.sqrt(np.nanmean((np.ma.masked_invalid(data_vector_selected))**2))
        else:
            antarctic_analysis = [0, [np.nan, np.nan], np.nan, np.nan, np.nan, np.nan,]
            antarctic_rmse = np.nan
                
        # make netCDF
        nc = Dataset(output_file, "a")
        coastal_grp = nc.createGroup(f"coastal_{var_name}")
        coastal_grp.createDimension("x", 1)        
        nobs = coastal_grp.createVariable("nobs", "i8", "x")
        nobs[:] = coastal_analysis[0]
        minval = coastal_grp.createVariable("min", "f8", "x")
        minval[:] = coastal_analysis[1][0]
        maxval = coastal_grp.createVariable("max", "f8", "x")
        maxval[:] = coastal_analysis[1][1]
        meanval = coastal_grp.createVariable("mean", "f8", "x")
        meanval[:] = coastal_analysis[2]
        variance = coastal_grp.createVariable("variance", "f8", "x")
        variance[:] = coastal_analysis[3]
        skewness = coastal_grp.createVariable("skewness", "f8", "x")
        skewness[:] = coastal_analysis[4]
        kurtosis = coastal_grp.createVariable("kurtosis", "f8", "x")
        kurtosis[:] = coastal_analysis[5]
        rmse = coastal_grp.createVariable("rmse", "f8", "x")
        rmse[:] = coastal_rmse
    
    
        offshore_highvar_grp = nc.createGroup(f"offshore_highvar_{var_name}")
        offshore_highvar_grp.createDimension("x", 1)
        nobs = offshore_highvar_grp.createVariable("nobs", "i8", "x")
        nobs[:] = offshore_highvar_analysis[0]
        minval = offshore_highvar_grp.createVariable("min", "f8", "x")
        minval[:] = offshore_highvar_analysis[1][0]
        maxval = offshore_highvar_grp.createVariable("max", "f8", "x")
        maxval[:] = offshore_highvar_analysis[1][1]
        meanval = offshore_highvar_grp.createVariable("mean", "f8", "x")
        meanval[:] = offshore_highvar_analysis[2]
        variance = offshore_highvar_grp.createVariable("variance", "f8", "x")
        variance[:] = offshore_highvar_analysis[3]
        skewness = offshore_highvar_grp.createVariable("skewness", "f8", "x")
        skewness[:] = offshore_highvar_analysis[4]
        kurtosis = offshore_highvar_grp.createVariable("kurtosis", "f8", "x")
        kurtosis[:] = offshore_highvar_analysis[5]
        rmse = offshore_highvar_grp.createVariable("rmse", "f8", "x")
        rmse[:] = offshore_highvar_rmse
    
        offshore_lowvar_grp = nc.createGroup(f"offshore_lowvar_{var_name}")
        offshore_lowvar_grp.createDimension("x", 1)
        nobs = offshore_lowvar_grp.createVariable("nobs", "i8", "x")
        nobs[:] = offshore_lowvar_analysis[0]
        minval = offshore_lowvar_grp.createVariable("min", "f8", "x")
        minval[:] = offshore_lowvar_analysis[1][0]
        maxval = offshore_lowvar_grp.createVariable("max", "f8", "x")
        maxval[:] = offshore_lowvar_analysis[1][1]
        meanval = offshore_lowvar_grp.createVariable("mean", "f8", "x")
        meanval[:] = offshore_lowvar_analysis[2]
        variance = offshore_lowvar_grp.createVariable("variance", "f8", "x")
        variance[:] = offshore_lowvar_analysis[3]
        skewness = offshore_lowvar_grp.createVariable("skewness", "f8", "x")
        skewness[:] = offshore_lowvar_analysis[4]
        kurtosis = offshore_lowvar_grp.createVariable("kurtosis", "f8", "x")
        kurtosis[:] = offshore_lowvar_analysis[5]
        rmse = offshore_lowvar_grp.createVariable("rmse", "f8", "x")
        rmse[:] = offshore_lowvar_rmse
    
        equatorial_grp = nc.createGroup(f"equatorial_band_{var_name}")
        equatorial_grp.createDimension("x", 1)
        nobs = equatorial_grp.createVariable("nobs", "i8", "x")
        nobs[:] = equatorial_analysis[0]
        minval = equatorial_grp.createVariable("min", "f8", "x")
        minval[:] = equatorial_analysis[1][0]
        maxval = equatorial_grp.createVariable("max", "f8", "x")
        maxval[:] = equatorial_analysis[1][1]
        meanval = equatorial_grp.createVariable("mean", "f8", "x")
        meanval[:] = equatorial_analysis[2]
        variance = equatorial_grp.createVariable("variance", "f8", "x")
        variance[:] = equatorial_analysis[3]
        skewness = equatorial_grp.createVariable("skewness", "f8", "x")
        skewness[:] = equatorial_analysis[4]
        kurtosis = equatorial_grp.createVariable("kurtosis", "f8", "x")
        kurtosis[:] = equatorial_analysis[5]
        rmse = equatorial_grp.createVariable("rmse", "f8", "x")
        rmse[:] = equatorial_rmse
    
        arctic_grp = nc.createGroup(f"arctic_{var_name}")
        arctic_grp.createDimension("x", 1)
        nobs = arctic_grp.createVariable("nobs", "i8", "x")
        nobs[:] = arctic_analysis[0]
        minval = arctic_grp.createVariable("min", "f8", "x")
        minval[:] = arctic_analysis[1][0]
        maxval = arctic_grp.createVariable("max", "f8", "x")
        maxval[:] = arctic_analysis[1][1]
        meanval = arctic_grp.createVariable("mean", "f8", "x")
        meanval[:] = arctic_analysis[2]
        variance = arctic_grp.createVariable("variance", "f8", "x")
        variance[:] = arctic_analysis[3]
        skewness = arctic_grp.createVariable("skewness", "f8", "x")
        skewness[:] = arctic_analysis[4]
        kurtosis = arctic_grp.createVariable("kurtosis", "f8", "x")
        kurtosis[:] = arctic_analysis[5]
        rmse = arctic_grp.createVariable("rmse", "f8", "x")
        rmse[:] = arctic_rmse
    
        antarctic_grp = nc.createGroup(f"antarctic_{var_name}")
        antarctic_grp.createDimension("x", 1)
        nobs = antarctic_grp.createVariable("nobs", "i8", "x")
        nobs[:] = antarctic_analysis[0]
        minval = antarctic_grp.createVariable("min", "f8", "x")
        minval[:] = antarctic_analysis[1][0]
        maxval = antarctic_grp.createVariable("max", "f8", "x")
        maxval[:] = antarctic_analysis[1][1]
        meanval = antarctic_grp.createVariable("mean", "f8", "x")
        meanval[:] = antarctic_analysis[2]
        variance = antarctic_grp.createVariable("variance", "f8", "x")
        variance[:] = antarctic_analysis[3]
        skewness = antarctic_grp.createVariable("skewness", "f8", "x")
        skewness[:] = antarctic_analysis[4]
        kurtosis = antarctic_grp.createVariable("kurtosis", "f8", "x")
        kurtosis[:] = antarctic_analysis[5]
        rmse = antarctic_grp.createVariable("rmse", "f8", "x")
        rmse[:] = antarctic_rmse
    
        nc.close()
    
    
def bin_data_uv(ds, output_file, lon_out=np.arange(0, 360, 1), lat_out=np.arange(-90, 90, 1), freq_out='1D'):
    
    binning = pyinterp.Binning2D(pyinterp.Axis(lon_out, is_circle=True),
                                 pyinterp.Axis(lat_out))
    
    
    # All spatial scale U    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['EWCT']).compute()
    mean_drifter_u = binning.variable('mean')
    variance_drifter_u = binning.variable('variance')
    timeserie_mean_drifter_u = (ds['EWCT'].resample(time=freq_out)).mean()
    timeserie_variance_drifter_u = (ds['EWCT'].resample(time=freq_out)).var()
    timeserie_rms_drifter_u = np.sqrt(((ds['EWCT']**2).resample(time=freq_out)).mean())
    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['ugos_interpolated']).compute()
    mean_map_ugos = binning.variable('mean')
    variance_map_ugos = binning.variable('variance')
    timeserie_mean_map_ugos = (ds['ugos_interpolated'].resample(time=freq_out)).mean()
    timeserie_variance_map_ugos = (ds['ugos_interpolated'].resample(time=freq_out)).var()
    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['mapping_err_u']).compute()
    mean_mapping_err_u = binning.variable('mean')
    variance_mapping_err_u = binning.variable('variance')
    timeserie_mean_mapping_err_u = (ds['mapping_err_u'].resample(time=freq_out)).mean()
    timeserie_variance_mapping_err_u = (ds['mapping_err_u'].resample(time=freq_out)).var()
    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['mapping_err_u']**2).compute()
    rmse_u = np.sqrt(binning.variable('mean'))
    timeserie_rmse_u = np.sqrt(((ds['mapping_err_u']**2).resample(time=freq_out)).mean())
    
    
    # All spatial scale V    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['NSCT']).compute()
    mean_drifter_v = binning.variable('mean')
    variance_drifter_v = binning.variable('variance')
    timeserie_mean_drifter_v = (ds['NSCT'].resample(time=freq_out)).mean()
    timeserie_variance_drifter_v = (ds['NSCT'].resample(time=freq_out)).var()
    timeserie_rms_drifter_v = np.sqrt(((ds['NSCT']**2).resample(time=freq_out)).mean())
    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['vgos_interpolated']).compute()
    mean_map_vgos = binning.variable('mean')
    variance_map_vgos = binning.variable('variance')
    timeserie_mean_map_vgos = (ds['vgos_interpolated'].resample(time=freq_out)).mean()
    timeserie_variance_map_vgos = (ds['vgos_interpolated'].resample(time=freq_out)).var()
    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['mapping_err_v']).compute()
    mean_mapping_err_v = binning.variable('mean')
    variance_mapping_err_v = binning.variable('variance')
    timeserie_mean_mapping_err_v = (ds['mapping_err_v'].resample(time=freq_out)).mean()
    timeserie_variance_mapping_err_v = (ds['mapping_err_v'].resample(time=freq_out)).var()
    
    binning.clear()
    binning = binning.push_delayed(ds.longitude, ds.latitude, ds['mapping_err_v']**2).compute()
    rmse_v = np.sqrt(binning.variable('mean'))
    timeserie_rmse_v = np.sqrt(((ds['mapping_err_v']**2).resample(time=freq_out)).mean())
    
    
    ds1 = xr.Dataset({'mean_drifter_u' : (('lat', 'lon'), mean_drifter_u.T), 
                      'variance_drifter_u' : (('lat', 'lon'), variance_drifter_u.T),
                      'mean_map_ugos' : (('lat', 'lon'), mean_map_ugos.T), 
                      'variance_map_ugos' : (('lat', 'lon'), variance_map_ugos.T),
                      'mean_mapping_err_u' : (('lat', 'lon'), mean_mapping_err_u.T), 
                      'variance_mapping_err_u' : (('lat', 'lon'), variance_mapping_err_u.T),
                      'rmse_u' : (('lat', 'lon'), rmse_u.T),
                      
                      'timeserie_mean_drifter_u' : (('time'), timeserie_mean_drifter_u.data), 
                      'timeserie_variance_drifter_u' : (('time'), timeserie_variance_drifter_u.data),
                      'timeserie_rms_drifter_u' : (('time'), timeserie_rms_drifter_u.data),
                      'timeserie_mean_map_ugos' : (('time'), timeserie_mean_map_ugos.data), 
                      'timeserie_variance_map_ugos' : (('time'), timeserie_variance_map_ugos.data),
                      'timeserie_mean_mapping_err_u' : (('time'), timeserie_mean_mapping_err_u.data), 
                      'timeserie_variance_mapping_err_u' : (('time'), timeserie_variance_mapping_err_u.data),
                      'timeserie_rmse_u' : (('time'), timeserie_rmse_u.data),
                      
                      'mean_drifter_v' : (('lat', 'lon'), mean_drifter_v.T), 
                      'variance_drifter_v' : (('lat', 'lon'), variance_drifter_v.T),
                      'mean_map_vgos' : (('lat', 'lon'), mean_map_vgos.T), 
                      'variance_map_vgos' : (('lat', 'lon'), variance_map_vgos.T),
                      'mean_mapping_err_v' : (('lat', 'lon'), mean_mapping_err_v.T), 
                      'variance_mapping_err_v' : (('lat', 'lon'), variance_mapping_err_v.T),
                      'rmse_v' : (('lat', 'lon'), rmse_v.T),
                      
                      'timeserie_mean_drifter_v' : (('time'), timeserie_mean_drifter_v.data), 
                      'timeserie_variance_drifter_v' : (('time'), timeserie_variance_drifter_v.data),
                      'timeserie_rms_drifter_v' : (('time'), timeserie_rms_drifter_v.data),
                      'timeserie_mean_map_vgos' : (('time'), timeserie_mean_map_vgos.data), 
                      'timeserie_variance_map_vgos' : (('time'), timeserie_variance_map_vgos.data),
                      'timeserie_mean_mapping_err_v' : (('time'), timeserie_mean_mapping_err_v.data), 
                      'timeserie_variance_mapping_err_v' : (('time'), timeserie_variance_mapping_err_v.data),
                      'timeserie_rmse_v' : (('time'), timeserie_rmse_v.data),
                      
                      },
                      coords={'lon': lon_out, 
                              'lat': lat_out,
                              'time': timeserie_rmse_u['time'],
                               }
                       )
    
    
    ds1.to_netcdf(output_file, group="all_scale", format="NETCDF4")

    
def compute_stat_scores_uv_by_regimes(ds_interp, output_file): 
    
    # distance_to_nearest_coast = '/home/ad/ballarm/data/wisa/data_challenge_ose//data/sad/distance_to_nearest_coastline_60.nc'
    # land_sea_mask = '/home/ad/ballarm/data/wisa/data_challenge_ose//data/sad/land_water_mask_60.nc'
    # variance_ssh = '/home/ad/ballarm/data/wisa/data_challenge_ose//data/sad/variance_cmems_dt_allsat.nc'
    distance_to_nearest_coast = '../data/sad/distance_to_nearest_coastline_60.nc'
    land_sea_mask = '../data/wisa/data_challenge_ose//data/sad/land_water_mask_60.nc'
    variance_ssh = '../data/wisa/data_challenge_ose//data/sad/variance_cmems_dt_allsat.nc'
    variance_criteria = 0.02             # min variance contour in m**2 to define the high variability regions
    coastal_distance_criteria = 200.     # max distance to coast in km to define the coastal regions
    
    lon_vector = ds_interp['longitude'].values
    lat_vector = ds_interp['latitude'].values
        
    # interpolate distance_to_nearest_coast, land_sea_mask, variance_ssh to lon/lat alongtrack
    ds = xr.open_dataset(land_sea_mask)
    x_axis = pyinterp.Axis(ds['lon'][:], is_circle=True)
    y_axis = pyinterp.Axis(ds['lat'][:])
    lsm = ds.variables["mask"][:].T
    
    lsm = np.where(lsm > 1, 1.0, lsm)
    # The undefined values must be set to nan.
    # lsm[lsm.mask] = float("nan")
    grid = pyinterp.Grid2D(x_axis, y_axis, lsm)
    lsm_interp = pyinterp.bivariate(grid, lon_vector, lat_vector, interpolator='nearest').reshape(lon_vector.shape)
    #plt.scatter(lon_vector[:10000], lat_vector[:10000], c=lsm_interp[:10000], s=10)
    #plt.show()
    
    ds = xr.open_dataset(distance_to_nearest_coast)
    x_axis = pyinterp.Axis(ds['lon'][:], is_circle=True)
    y_axis = pyinterp.Axis(ds['lat'][:])
    distance = ds.variables["distance"][:].T
    # The undefined values must be set to nan.
    # distance[distance.mask] = float("nan")
    grid = pyinterp.Grid2D(x_axis, y_axis, distance.data)
    distance_interp = pyinterp.bivariate(grid, lon_vector, lat_vector).reshape(lon_vector.shape)
    
    ds = xr.open_dataset(variance_ssh)
    x_axis = pyinterp.Axis(ds['longitude'][:], is_circle=True)
    y_axis = pyinterp.Axis(ds['latitude'][:])
    variance = ds.variables["sla"][:].T
    # The undefined values must be set to nan.
    # distance[distance.mask] = float("nan")
    grid = pyinterp.Grid2D(x_axis, y_axis, variance.data)
    variance_interp = pyinterp.bivariate(grid, lon_vector, lat_vector).reshape(lon_vector.shape)
    
    # Clean with lsm
    msk_land_data = np.ma.masked_where(lsm_interp == 1, lsm_interp).mask
    msk_coastal_data = np.ma.masked_where(distance_interp <= coastal_distance_criteria, distance_interp).mask
    msk_offshore_data = np.ma.masked_where(distance_interp >= coastal_distance_criteria, distance_interp).mask
    msk_lowvar_data = np.ma.masked_where(variance_interp <= variance_criteria, variance_interp).mask
    msk_highvar_data = np.ma.masked_where(variance_interp >= variance_criteria, variance_interp).mask
    msk_extra_equatorial_band = np.ma.masked_where(np.abs(lat_vector) > 10, lat_vector).mask
    msk_arctic = np.ma.masked_where(lat_vector < 70., lat_vector).mask
    msk_antarctic = np.ma.masked_where(lat_vector > -70., lat_vector).mask
    
    for var_name in ['mapping_err_u', 'mapping_err_v', 'ugos_interpolated', 'EWCT', 'vgos_interpolated', 'NSCT']:
        data_vector = ds_interp[var_name].values
    
        # distance <= 200km
        msk = msk_land_data + msk_offshore_data
        data_vector_selected = np.ma.masked_where(msk, data_vector).compressed()
        if data_vector_selected.size > 0:
            coastal_analysis = stats.describe(data_vector_selected, nan_policy='omit')
            coastal_rmse = np.sqrt(np.nanmean((np.ma.masked_invalid(data_vector_selected))**2))
            #lon_vector_selected = np.ma.masked_where(msk, lon_vector).compressed()
            #lat_vector_selected = np.ma.masked_where(msk, lat_vector).compressed()
            #plt.scatter(lon_vector_selected, lat_vector_selected, c=data_vector_selected, s=10)
            #plt.show()
        else:
            coastal_analysis = [0, [np.nan, np.nan], np.nan, np.nan, np.nan, np.nan,]
            coastal_rmse = np.nan
    
        # distance >= 200km & variance >= 0.02
        msk = msk_land_data + msk_coastal_data + msk_lowvar_data
        data_vector_selected = np.ma.masked_where(msk, data_vector).compressed()
        if data_vector_selected.size > 0:
            offshore_highvar_analysis = stats.describe(data_vector_selected, nan_policy='omit')
            offshore_highvar_rmse = np.sqrt(np.nanmean((np.ma.masked_invalid(data_vector_selected))**2))
            #lon_vector_selected = np.ma.masked_where(msk, lon_vector).compressed()
            #lat_vector_selected = np.ma.masked_where(msk, lat_vector).compressed()
            #plt.scatter(lon_vector_selected, lat_vector_selected, c=data_vector_selected, s=10)
            #plt.show()
        else:
            offshore_highvar_analysis = [0, [np.nan, np.nan], np.nan, np.nan, np.nan, np.nan,]
            offshore_highvar_rmse = np.nan
    
        # distance >= 200km & variance <= 0.02
        msk = msk_land_data + msk_coastal_data + msk_highvar_data
        data_vector_selected = np.ma.masked_where(msk, data_vector).compressed()
        if data_vector_selected.size > 0:
            offshore_lowvar_analysis = stats.describe(data_vector_selected, nan_policy='omit')
            offshore_lowvar_rmse = np.sqrt(np.nanmean((np.ma.masked_invalid(data_vector_selected))**2))
            #lon_vector_selected = np.ma.masked_where(msk, lon_vector).compressed()
            #lat_vector_selected = np.ma.masked_where(msk, lat_vector).compressed()
            #plt.scatter(lon_vector_selected, lat_vector_selected, c=data_vector_selected, s=10)
            #plt.show()
        else:
            offshore_lowvar_analysis = [0, [np.nan, np.nan], np.nan, np.nan, np.nan, np.nan,]
            offshore_lowvar_rmse = np.nan
    
        # Equatorial band
        msk = msk_land_data + msk_extra_equatorial_band
        data_vector_selected = np.ma.masked_where(msk, data_vector).compressed()
        if data_vector_selected.size > 0:
            equatorial_analysis = stats.describe(data_vector_selected, nan_policy='omit')
            equatorial_rmse = np.sqrt(np.nanmean((np.ma.masked_invalid(data_vector_selected))**2))
        else:
            equatorial_analysis = [0, [np.nan, np.nan], np.nan, np.nan, np.nan, np.nan,]
            equatorial_rmse = np.nan
    
        # Arctic
        msk = msk_land_data + msk_arctic
        data_vector_selected = np.ma.masked_where(msk, data_vector).compressed()
        if data_vector_selected.size > 0:
            arctic_analysis = stats.describe(data_vector_selected, nan_policy='omit')
            arctic_rmse = np.sqrt(np.nanmean((np.ma.masked_invalid(data_vector_selected))**2))
        else:
            arctic_analysis = [0, [np.nan, np.nan], np.nan, np.nan, np.nan, np.nan,]
            arctic_rmse = np.nan
        
        # AntArctic
        msk = msk_land_data + msk_antarctic
        data_vector_selected = np.ma.masked_where(msk, data_vector).compressed()
        if data_vector_selected.size > 0:
            antarctic_analysis = stats.describe(data_vector_selected, nan_policy='omit')
            antarctic_rmse = np.sqrt(np.nanmean((np.ma.masked_invalid(data_vector_selected))**2))
        else:
            antarctic_analysis = [0, [np.nan, np.nan], np.nan, np.nan, np.nan, np.nan,]
            antarctic_rmse = np.nan
                
        # make netCDF
        nc = Dataset(output_file, "a")
        coastal_grp = nc.createGroup(f"coastal_{var_name}")
        coastal_grp.createDimension("x", 1)        
        nobs = coastal_grp.createVariable("nobs", "i8", "x")
        nobs[:] = coastal_analysis[0]
        minval = coastal_grp.createVariable("min", "f8", "x")
        minval[:] = coastal_analysis[1][0]
        maxval = coastal_grp.createVariable("max", "f8", "x")
        maxval[:] = coastal_analysis[1][1]
        meanval = coastal_grp.createVariable("mean", "f8", "x")
        meanval[:] = coastal_analysis[2]
        variance = coastal_grp.createVariable("variance", "f8", "x")
        variance[:] = coastal_analysis[3]
        skewness = coastal_grp.createVariable("skewness", "f8", "x")
        skewness[:] = coastal_analysis[4]
        kurtosis = coastal_grp.createVariable("kurtosis", "f8", "x")
        kurtosis[:] = coastal_analysis[5]
        rmse = coastal_grp.createVariable("rmse", "f8", "x")
        rmse[:] = coastal_rmse
    
    
        offshore_highvar_grp = nc.createGroup(f"offshore_highvar_{var_name}")
        offshore_highvar_grp.createDimension("x", 1)
        nobs = offshore_highvar_grp.createVariable("nobs", "i8", "x")
        nobs[:] = offshore_highvar_analysis[0]
        minval = offshore_highvar_grp.createVariable("min", "f8", "x")
        minval[:] = offshore_highvar_analysis[1][0]
        maxval = offshore_highvar_grp.createVariable("max", "f8", "x")
        maxval[:] = offshore_highvar_analysis[1][1]
        meanval = offshore_highvar_grp.createVariable("mean", "f8", "x")
        meanval[:] = offshore_highvar_analysis[2]
        variance = offshore_highvar_grp.createVariable("variance", "f8", "x")
        variance[:] = offshore_highvar_analysis[3]
        skewness = offshore_highvar_grp.createVariable("skewness", "f8", "x")
        skewness[:] = offshore_highvar_analysis[4]
        kurtosis = offshore_highvar_grp.createVariable("kurtosis", "f8", "x")
        kurtosis[:] = offshore_highvar_analysis[5]
        rmse = offshore_highvar_grp.createVariable("rmse", "f8", "x")
        rmse[:] = offshore_highvar_rmse
    
        offshore_lowvar_grp = nc.createGroup(f"offshore_lowvar_{var_name}")
        offshore_lowvar_grp.createDimension("x", 1)
        nobs = offshore_lowvar_grp.createVariable("nobs", "i8", "x")
        nobs[:] = offshore_lowvar_analysis[0]
        minval = offshore_lowvar_grp.createVariable("min", "f8", "x")
        minval[:] = offshore_lowvar_analysis[1][0]
        maxval = offshore_lowvar_grp.createVariable("max", "f8", "x")
        maxval[:] = offshore_lowvar_analysis[1][1]
        meanval = offshore_lowvar_grp.createVariable("mean", "f8", "x")
        meanval[:] = offshore_lowvar_analysis[2]
        variance = offshore_lowvar_grp.createVariable("variance", "f8", "x")
        variance[:] = offshore_lowvar_analysis[3]
        skewness = offshore_lowvar_grp.createVariable("skewness", "f8", "x")
        skewness[:] = offshore_lowvar_analysis[4]
        kurtosis = offshore_lowvar_grp.createVariable("kurtosis", "f8", "x")
        kurtosis[:] = offshore_lowvar_analysis[5]
        rmse = offshore_lowvar_grp.createVariable("rmse", "f8", "x")
        rmse[:] = offshore_lowvar_rmse
    
        equatorial_grp = nc.createGroup(f"equatorial_band_{var_name}")
        equatorial_grp.createDimension("x", 1)
        nobs = equatorial_grp.createVariable("nobs", "i8", "x")
        nobs[:] = equatorial_analysis[0]
        minval = equatorial_grp.createVariable("min", "f8", "x")
        minval[:] = equatorial_analysis[1][0]
        maxval = equatorial_grp.createVariable("max", "f8", "x")
        maxval[:] = equatorial_analysis[1][1]
        meanval = equatorial_grp.createVariable("mean", "f8", "x")
        meanval[:] = equatorial_analysis[2]
        variance = equatorial_grp.createVariable("variance", "f8", "x")
        variance[:] = equatorial_analysis[3]
        skewness = equatorial_grp.createVariable("skewness", "f8", "x")
        skewness[:] = equatorial_analysis[4]
        kurtosis = equatorial_grp.createVariable("kurtosis", "f8", "x")
        kurtosis[:] = equatorial_analysis[5]
        rmse = equatorial_grp.createVariable("rmse", "f8", "x")
        rmse[:] = equatorial_rmse
    
        arctic_grp = nc.createGroup(f"arctic_{var_name}")
        arctic_grp.createDimension("x", 1)
        nobs = arctic_grp.createVariable("nobs", "i8", "x")
        nobs[:] = arctic_analysis[0]
        minval = arctic_grp.createVariable("min", "f8", "x")
        minval[:] = arctic_analysis[1][0]
        maxval = arctic_grp.createVariable("max", "f8", "x")
        maxval[:] = arctic_analysis[1][1]
        meanval = arctic_grp.createVariable("mean", "f8", "x")
        meanval[:] = arctic_analysis[2]
        variance = arctic_grp.createVariable("variance", "f8", "x")
        variance[:] = arctic_analysis[3]
        skewness = arctic_grp.createVariable("skewness", "f8", "x")
        skewness[:] = arctic_analysis[4]
        kurtosis = arctic_grp.createVariable("kurtosis", "f8", "x")
        kurtosis[:] = arctic_analysis[5]
        rmse = arctic_grp.createVariable("rmse", "f8", "x")
        rmse[:] = arctic_rmse
    
        antarctic_grp = nc.createGroup(f"antarctic_{var_name}")
        antarctic_grp.createDimension("x", 1)
        nobs = antarctic_grp.createVariable("nobs", "i8", "x")
        nobs[:] = antarctic_analysis[0]
        minval = antarctic_grp.createVariable("min", "f8", "x")
        minval[:] = antarctic_analysis[1][0]
        maxval = antarctic_grp.createVariable("max", "f8", "x")
        maxval[:] = antarctic_analysis[1][1]
        meanval = antarctic_grp.createVariable("mean", "f8", "x")
        meanval[:] = antarctic_analysis[2]
        variance = antarctic_grp.createVariable("variance", "f8", "x")
        variance[:] = antarctic_analysis[3]
        skewness = antarctic_grp.createVariable("skewness", "f8", "x")
        skewness[:] = antarctic_analysis[4]
        kurtosis = antarctic_grp.createVariable("kurtosis", "f8", "x")
        kurtosis[:] = antarctic_analysis[5]
        rmse = antarctic_grp.createVariable("rmse", "f8", "x")
        rmse[:] = antarctic_rmse
    
        nc.close()
        
        
    
def compute_stat_scores_uv(ds_interp, output_file):
     
    logging.info("Compute mapping error all scales")
    ds_interp['mapping_err_u'] = ds_interp['ugos_interpolated'] - ds_interp['EWCT']
    ds_interp['mapping_err_v'] = ds_interp['vgos_interpolated'] - ds_interp['NSCT']
    
    logging.info("Compute statistics")
    # Bin data maps
    bin_data_uv(ds_interp, output_file)
    logging.info("Stat file saved as: %s", output_file)
    
    logging.info("Compute statistics by oceanic regime")
    compute_stat_scores_uv_by_regimes(ds_interp, output_file)
 
    
