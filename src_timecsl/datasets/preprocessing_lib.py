import warnings
import numpy as np
import math
from skimage.restoration import denoise_wavelet
import pandas as pd


def Upsampling_array(X, sampling : int = 1, method : str = 'mean'):
    finale = np.zeros((np.shape(X)[0],np.shape(X)[1],np.shape(X)[2]//(math.ceil(sampling))))
    for i_eme in range(np.shape(X)[0]):
        for k in range(np.shape(X)[1]):
            if method=='mean':
                finale[i_eme,k,:] = [X[i_eme,k,:][i:math.ceil(sampling)+i].mean() for i in range(0,np.size(X[i_eme,k,:]),math.ceil(sampling))]
            elif method=='max':
                finale[i_eme,k,:] = [X[i_eme,k,:][i:math.ceil(sampling)+i].max() for i in range(0,np.size(X[i_eme,k,:]),math.ceil(sampling))]
            else:
                raise Exception('method not exist')
    return finale


def replace_nans(aggregated: np.array, disaggregated: np.array):
    if aggregated is not None:
        aggregated = pd.DataFrame(np.array(aggregated))
        #aggregated = 
        aggregated.fillna(0, inplace=True)
        aggregated = aggregated.values

    if disaggregated is not None:
        disaggregated = pd.DataFrame(np.array(disaggregated))  
        disaggregated.fillna(0, inplace=True)
        disaggregated = disaggregated.values

    return aggregated, disaggregated


def replace_nans_interpolation(aggregated: np.array = None, disaggregated: np.array = None):
    if aggregated is not None:
        aggregated = pd.DataFrame(np.array(aggregated))
        aggregated.interpolate(method='linear', limit_direction='forward', inplace=True)
        aggregated = aggregated.values
        
    if disaggregated is not None:
        disaggregated = pd.DataFrame(np.array(disaggregated))   
        disaggregated.interpolate(method='linear', limit_direction='forward', inplace=True)
        disaggregated = disaggregated.values

    return aggregated, disaggregated


def normalize_sequences(aggregated: np.array  = None, disaggregated: np.array  = None, mmax: float  = None, threshould: float  = None):
    if aggregated is not None:
        aggregated = pd.DataFrame(aggregated)
        if mmax is None:
            mmax = aggregated.max(axis=0) # Normalisation par rapport à chaque features [activate, reactive, apparent]/ [Max_activate, Max_reactive, Max_apparent]
         
        aggregated = aggregated / mmax  
        aggregated = aggregated.values

    if disaggregated is not None:
        disaggregated = pd.DataFrame(disaggregated)
        disaggregated = disaggregated / mmax[0] # Normalisation par rapport à l'activate uniquement comme c'est l'output
        disaggregated = disaggregated.values

    threshould_normalized = threshould/mmax
    return aggregated, disaggregated, threshould_normalized

    
def standardize_sequences(aggregated: np.array  = None, disaggregated: np.array  = None, mains_mean: float  = None,
                       mains_std: float  = None, meter_mean: float  = None, meter_std: float  = None):
    # I have to modify this for case where a sequence contain only zeros as values
         ######
    # TODO: If sequence is a bad (all zeros) then means/stds will be problematic
    ######
    if aggregated is not None:
        aggregated = pd.DataFrame(np.array(aggregated))
        if mains_mean is None and mains_std is None:
            mains_mean = aggregated.mean()
            mains_std = aggregated.std()
            
        aggregated = (aggregated - mains_mean)/mains_std
        aggregated = aggregated.values

    if disaggregated is not None:
        disaggregated = pd.DataFrame(np.array(disaggregated))
        if meter_mean is None and meter_std is None:
            meter_mean = disaggregated.mean()
            meter_std = disaggregated.std()
        disaggregated = (disaggregated-meter_mean)/meter_std
        disaggregated = disaggregated.values

    return aggregated, disaggregated

def denoise(aggregated: np.array, disaggregated: np.array):
    
    aggregated = denoise_wavelet(aggregated, wavelet='haar', wavelet_levels=3)
    disaggregated = denoise_wavelet(disaggregated, wavelet='haar', wavelet_levels=3)
    return aggregated, disaggregated


def add_gaussian_noise(aggregated: np.array, noise_factor: float = 0.1):
    noise = noise_factor * np.random.normal(0, 1, aggregated.shape)
    aggregated = aggregated + noise
    return aggregated

def select_sequences(aggregate, disaggregate,threshould_normalized):
    aggregate_active,disaggregate_active=[],[]
    for i in range(np.shape(aggregate)[0]):
        if not is_bad_sequence(disaggregate[i],threshould_normalized):
            aggregate_active.append(aggregate[i]), 
            disaggregate_active.append(disaggregate[i])
            
    aggregate_active = np.array(aggregate_active)
    disaggregate_active = np.array(disaggregate_active)                                                                                                                                                                                                                                        
    return aggregate_active, disaggregate_active

def is_bad_sequence(chunk: np.array,threshould_normalized: float):
    return (chunk > threshould_normalized).all()

def align_sequences(aggregated: np.array, disaggregated: np.array):
    aggregated = aggregated[~aggregated.index.duplicated()]
    disaggregated = disaggregated[~disaggregated.index.duplicated()]
    ix = aggregated.index.intersection(disaggregated.index)
    aggregated = aggregated[ix]
    disaggregated = disaggregated[ix]
    return aggregated, disaggregated

def replace_with_zero_small_values(aggregated: np.array, disaggregated: np.array, threshold: int):
    aggregated[aggregated < threshold] = 0
    disaggregated[disaggregated < threshold] = 0
    return aggregated, disaggregated