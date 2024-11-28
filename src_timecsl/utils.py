import multiprocessing as mp
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from dateutil.relativedelta import relativedelta
from functools import partial
from itertools import cycle
from multiprocessing import Pool

from pyitlib import discrete_random_variable as drv




def estimate_required_time(nb_items_in_list, current_index, time_elapsed, interval=100):
    current_index += 1  # increment current_idx by 1
    if current_index % interval == 0 or current_index == nb_items_in_list:
        # make time estimation and put to string format
        seconds = (nb_items_in_list - current_index) * (time_elapsed / current_index)
        time_estimation = relativedelta(seconds=int(seconds))
        time_estimation_string = f'{time_estimation.hours:02}:{time_estimation.minutes:02}:{time_estimation.seconds:02}'

        # extract progress bar
        progress_bar = prog_bar(i=current_index, n=nb_items_in_list)

        # display info
        if current_index == nb_items_in_list:
            sys.stdout.write(f'\r{progress_bar} -- estimated required time = {time_estimation_string} -- Finished!')
        else:
            sys.stdout.write(f'\r{progress_bar} -- estimated required time = {time_estimation_string}')



def get_score(items, metric, factors_codes_dataset):
    ''' Compute metric score on a specific factors-codes representation
    
    :param items:                      representation seed and run seed
    :param metric:                     metric to use to compute score
    :param factors_codes_dataset:      function to create factors-codes dataset
    '''
    # extract seeds
    representation_seed, run_seed = items
    
    # create factors-codes dataset
    np.random.seed(representation_seed)
    factors, codes = factors_codes_dataset()
    
    # compute score
    np.random.seed(run_seed)
    score = metric(factors=factors, codes=codes)
    
    return score


def get_experiment_seeds(nb_representations, nb_runs):
    ''' Extract all seeds to use in the experiment
    
    :param nb_representations:      number of random representations to generate
    :param nb_runs:                 number of times we run the metric on the same random representation
    '''
    # seeds corresponding to different random representations
    repr_seeds = [repr_seed for repr_seed in range(nb_representations)]
    
    # seeds corresponding to the experiment runs
    # each pair of representation/run has a unique seed
    # it allows to take into account the stochasticity of the metrics
    run_seeds = [run_seed for run_seed in range(nb_representations * nb_runs)]
    
    # combine representation seeds with their corresponding run seeds
    seeds = [(repr_seed, run_seed)
             for repr_idx, repr_seed in enumerate(repr_seeds)
             for run_idx, run_seed in enumerate(run_seeds)
             if repr_idx * nb_runs <= run_idx < (repr_idx + 1) * nb_runs]
    
    return seeds


def get_bin_index(x, nb_bins):
    ''' Discretize input variable
    
    :param x:           input variable
    :param nb_bins:     number of bins to use for discretization
    '''
    # get bins limits
    bins = np.linspace(0, 1, nb_bins + 1)

    # discretize input variable
    return np.digitize(x, bins[:-1], right=False).astype(int)


def get_mutual_information(x, y, normalize=True):
    ''' Compute mutual information between two random variables
    
    :param x:      random variable
    :param y:      random variable
    '''
    if normalize:
        return drv.information_mutual_normalised(x, y, norm_factor='Y', cartesian_product=True)
    else:
        return drv.information_mutual(x, y, cartesian_product=True)


def get_artificial_factors_dataset(nb_examples, nb_factors, distribution, dist_kwargs):
    # initialize factors dataset
    factors = np.zeros((nb_examples, nb_factors))
    
    # fill array with random continuous factors values from the distribution
    for line_idx in range(0, nb_examples):
        for column_idx in range(0, nb_factors):
            factor_value = distribution(**dist_kwargs)
            factors[line_idx, column_idx] = factor_value
    
    return factors
    

def get_nb_jobs(n_jobs):
    """ Return the number of parallel jobs specified by n_jobs

    :param n_jobs:      the number of jobs the user want to use in parallel

    :return: the number of parallel jobs
    """
    # set nb_jobs to max by default
    nb_jobs = mp.cpu_count()

    if n_jobs != 'max':
        if int(n_jobs) > mp.cpu_count():
            print(f'Max number of parallel jobs is "{mp.cpu_count()}" but received "{int(n_jobs)}" -- '
                  f'setting nb of parallel jobs to {nb_jobs}')
        else:
            nb_jobs = int(n_jobs)

    return nb_jobs


def launch_multi_process(iterable, func, n_jobs, chunksize=1, ordered=True, timer_verbose=True, interval=100, **kwargs):
    """ Calls function using multi-processing pipes
        https://guangyuwu.wordpress.com/2018/01/12/python-differences-between-imap-imap_unordered-and-map-map_async/

    :param iterable:        items to process with function func
    :param func:            function to multi-process
    :param n_jobs:          number of parallel jobs to use
    :param chunksize:       size of chunks given to each worker
    :param ordered:         True: iterable is returned while still preserving the ordering of the input iterable
                            False: iterable is returned regardless of the order of the input iterable -- better perf
    :param timer_verbose:   display time estimation when set to True
    :param interval:        estimate remaining time when (current_index % interval) == 0
    :param kwargs:          additional keyword arguments taken by function func

    :return: function outputs
    """
    # define pool of workers
    pool = Pool(processes=n_jobs)

    # define partial function and pool function
    func = partial(func, **kwargs)
    pool_func = pool.imap if ordered else pool.imap_unordered

    # initialize variables
    func_returns = []
    nb_items_in_list = len(iterable) if timer_verbose else None
    start = time.time() if timer_verbose else None

    # iterate over iterable
    for i, func_return in enumerate(pool_func(func, iterable, chunksize=chunksize)):
        # store function output
        func_returns.append(func_return)

        # compute remaining time
        if timer_verbose:
            estimate_required_time(nb_items_in_list=nb_items_in_list, current_index=i,
                                   time_elapsed=time.time() - start, interval=interval)
    if timer_verbose:
        sys.stdout.write('\n')

    # wait for all worker to finish and close the pool
    pool.close()
    pool.join()

    return func_returns


def prog_bar(i, n, bar_size=16):
    """ Create a progress bar to estimate remaining time

    :param i:           current iteration
    :param n:           total number of iterations
    :param bar_size:    size of the bar

    :return: a visualisation of the progress bar
    """
    bar = ''
    done = (i * bar_size) // n

    for j in range(bar_size):
        bar += '█' if j <= done else '░'

    message = f'{bar} {i}/{n}'
    return message
