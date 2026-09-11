import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bambi as bmb
import arviz as az
az.rcParams['stats.ci_prob'] = .95
from os import listdir
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor

import logging
logger = logging.getLogger('pymc')
logger.setLevel(logging.ERROR)
pd.options.mode.chained_assignment = None  

from src.utils import import_config

#######################################################################
### Configs ###########################################################
#######################################################################

DATA_PATH = '../data'
RESULTS_PATH = '../results'
FIG_PATH = '../figs'

# Subjects to exclude based on preregistered criterion
IDS_TO_EXCLUDE = [
    929094, 297827, 835395, 123289, 941076, 786858, 209368, 385852,
    988204, 586980
    ]


#######################################################################
### Data Loading ######################################################
#######################################################################

def load_from_dir(path, na_values=[]):
    df = []
    for f in listdir(path):
        if '.csv' in f:
            df.append(pd.read_csv(f'{path}/{f}', na_values=na_values))
    df = pd.concat(df)
    return df

def info_from_fname(fname, field):
    '''
    Get information from fname using field key

    Arguments
    ---------
    fname : str
        File name
    field : str
        Field key to get value for
    
    Returns
    -------
    field_value : str
        Value for specified field
    '''
    key_vals = [
        (f.split('-')[0], '-'.join(f.split('-')[1:]))
        for f in fname.split('.')[0].split('_')
        ]
    fname_info = {key: val for key, val in key_vals}
    field_value = fname_info[field]
    return field_value

def load_subjects_condition(data_path):
    """
    Load subject-condition assignments.

    Arguments
    ---------
    data_path : str
        Path to data directory

    Returns
    -------
    subj_df : pd.DataFrame
        DataFrame with subject IDs and condition assignments
    """
    
    # Load data
    training_df = load_from_dir(f'{data_path}/training', na_values='null')

    # Exclude participants
    idx = np.isin(training_df['id'], IDS_TO_EXCLUDE, invert=True)
    training_df = training_df.loc[idx]
    training_df = training_df.reset_index(drop=True)

    # Code for semantic congruency condition
    training_df['sem_congruent'] = 1 - training_df['between_cond'].astype(int)

    # Get subject and condition information
    subj_df = training_df[['id', 'sem_congruent']].drop_duplicates()

    return subj_df

def load_transition_influence(results_path, data_path):
    """
    Load fit transition influence coefficients.

    Arguments
    ---------
    results_path : str
        Path to results directory
    data_path : str
        Path to data directory

    Returns
    -------
    trans_influence_df : pd.DataFrame
        Transition influence coefficients
    """


    trans_influence_df = []
    for fname in listdir(f'{results_path}/training/agent'):

        # Load trace
        with open(f'{results_path}/training/agent/{fname}', 'rb') as f:
            trace = pickle.load(f)

        # Extract coefficients
        coefs = az.summary(trace)['mean'].values
        coefs = coefs.reshape(-1, 2)

        # Add to full results set
        coefs_df = pd.DataFrame({
            'id': info_from_fname(fname, 'agent'),
            'action': [1, 2, 3],
            'spurious': coefs[:, 0],
            'causal': coefs[:, 1]
        })
        trans_influence_df.append(coefs_df)

    # Combine into one data frame
    trans_influence_df = pd.concat(trans_influence_df)
    trans_influence_df = trans_influence_df.reset_index(drop=True)

    # Format
    trans_influence_df['id'] = trans_influence_df['id'].astype(int)

    # Convert to long
    trans_influence_df = pd.melt(
        trans_influence_df,
        id_vars = ['id', 'action'],
        value_vars = ['spurious', 'causal'],
        var_name = 'transition',
        value_name = 'coef'
    )
    trans_influence_df = trans_influence_df.reset_index(drop=True)

    # Recode so spurious = 0 factor
    trans_influence_df['transition'] = pd.Categorical(
        trans_influence_df['transition'],
        categories = ['spurious', 'causal'],
        ordered = True
    )

    # Add condition information
    subj_df = load_subjects_condition(data_path)
    trans_influence_df = pd.merge(
        trans_influence_df,
        subj_df,
        on = 'id'
    )

    return trans_influence_df


#######################################################################
### Power Analysis ####################################################
#######################################################################

def sample_from_fit_model(trans_influence_df):
    """
    Sample parameter values from a fitted transition influence model.

    Arguments
    ----------
    trans_influence_df : pd.DataFrame
        Transition influence coefficients data frame.

    Returns
    -------
    posterior_draws : dict
        Dictionary of posterior draws for needed parameters.
    """

    print('Fitting initial model for posterior draws...')

    # Fit the transition influence Bayesian regression model 
    model = bmb.Model(
        'coef ~ sem_congruent*transition + (1|action) + (1|id)',
        trans_influence_df,
        family = 'gaussian',
        link = 'identity'
    )
    fit = model.fit(
        random_seed = 1,
        tune = 4000,
        draws = 2000,
        target_accept = .95
    )
    print(az.summary(fit))

    # Get posterior draws for parameters of interest for data simulation
    posterior = fit.posterior
    posterior_draws = {
        'b0': posterior['Intercept'].values.flatten(),
        'b_congruent': posterior['sem_congruent'].values.flatten(),
        'b_transition': posterior['transition'].values.flatten(),
        'b_interaction': posterior['sem_congruent:transition'].values.flatten(),
        'action_sd': posterior['1|action_sigma'].values.flatten(),
        'id_sd': posterior['1|id_sigma'].values.flatten(),
        'sigma': posterior['sigma'].values.flatten(),
    }

    return posterior_draws

def simulate_data(
        n_per_group,
        b0,
        b_congruent,
        b_transition,
        b_interaction,
        action_sd,
        id_sd,
        sigma,
        seed = None
):
    """
    Simulate data based on specified parameters.

    Arguments
    ----------
    n_per_group : int
        Number of participants per semantic congruency condition.
    b0 : float
        Intercept beta.
    b_congruent : float
        Semantic congruency (incongruent - 0 vs congruent - 1) beta.
    b_transition : float
        Transition type (spurious - 0 vs causal - 1) beta.
    b_interaction : float
        Semantic congruency by transition interaction beta.
    action_sd : float
        Standard deviation of random intercepts for predicted action.
    id_sd : float
        Standard deviation of random intercepts for participant.
    sigma : float
        Standard deviation of estimated residuals.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    df : pandas.DataFrame
        Simulated data.
    """
    rng = np.random.default_rng(seed)

    # Total number of participants from number per condition
    n_total = n_per_group*2
    sem_congruent = np.repeat([0, 1], n_per_group)

    # Sample random intercepts
    action_effects = rng.normal(0, action_sd, size=4)
    participant_effects = rng.normal(0, id_sd, size=n_total)

    # Simulate data
    rows = []
    for p in range(n_total):
        cong = sem_congruent[p]
        for transition in [0, 1]:
            for action in range(4):
                mu = (b0
                      + b_congruent*cong
                      + b_transition*transition
                      + b_interaction*cong*transition
                      + action_effects[action]
                      + participant_effects[p])
                coef = mu + rng.normal(0, sigma)
                rows.append([p, cong, transition, action, coef])

    # Format simulated data for model fitting
    df = pd.DataFrame(
        rows,
        columns = ['id', 'sem_congruent', 'transition', 'action', 'coef']
        )
    df['transition'] = pd.Categorical(
            df['transition'].map({0: 'spurious', 1: 'causal'}),
            categories = ['spurious', 'causal'],
            ordered = True
        )
    df['action'] = df['action'].astype(str)

    return df

def run_one_simulation(args):
    """
    Run one power-analysis simulation.

    Arguments
    ----------
    args : tuple
        Tuple of arguments for the simulation.

    Returns
    -------
    success : bool
        True if the HDI for the interaction effect does not contain 0,
        False otherwise.
    """
    (
        i,
        n_per_group,
        posterior_draws,
        hdi_prob,
        shrink_factor,
        draws,
        tune
    ) = args

    rng = np.random.default_rng(i)

    # Number of posterior draws available for sampling
    n_draws_available = len(posterior_draws['b_interaction'])

    # Draw parameter values from the posterior 
    idx = rng.integers(0, n_draws_available)  
    b0 = posterior_draws['b0'][idx]
    b_congruent = posterior_draws['b_congruent'][idx]
    b_transition = posterior_draws['b_transition'][idx]
    id_sd = posterior_draws['id_sd'][idx]
    action_sd = posterior_draws['action_sd'][idx]
    sigma = posterior_draws['sigma'][idx]

    # Draw the target interaction effect size, which can be 
    # shrunk by a specified factor to simulate smaller effect sizes
    b_interaction = posterior_draws['b_interaction'][idx]*shrink_factor

    # Simulate data
    df = simulate_data(
        n_per_group,
        b0,
        b_congruent,
        b_transition,
        b_interaction,
        action_sd,
        id_sd,
        sigma,
        seed = i
        )

    # Fit the model to the simulated data
    model = bmb.Model(
        'coef ~ sem_congruent*transition + (1|action) + (1|id)',
        df,
        family = 'gaussian',
        link = 'identity'
    )

    fit = model.fit(
        draws = draws,
        tune = tune,
        chains = 2,
        cores = 2,
        progressbar = False,
        random_seed = i
    )

     # Assess whether the HDI is greater than 0
    hdi = az.hdi(
        fit,
        var_names = ['sem_congruent:transition'],
        hdi_prob = hdi_prob
    )['sem_congruent:transition'].values[0]
    success = hdi[0] > 0

    return success

def run_power_sim_posterior(
        n_per_group,
        posterior_draws,
        n_sims = 300,
        draws = 500,
        tune = 500,
        hdi_prob = 0.95,
        shrink_factor = 1.0,
        rng_seed = 0,
        n_workers = 2
    ):
    print(
        f'Running for n_per_group = {n_per_group}, '
        f'shrink_factor = {shrink_factor}...'
    )

    # Create arguments for each simulation
    args = [
        (
            i,
            n_per_group,
            posterior_draws,
            hdi_prob,
            shrink_factor,
            draws,
            tune
        )
        for i in range(n_sims)
    ]

    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        successes = list(executor.map(run_one_simulation, args))

    # Power is the proportion of simulations where HDI > 0
    power = sum(successes)/n_sims

    print(f'    Finished: Power = {power:.2f}')

    return power

def plot_results(results, fname=False):
    """
    Plot power analysis results as a heatmap.

    Arguments
    ----------
    results : pd.DataFrame
        Power analysis results.
    fname : str
        File name to save the figure. If False, the figure is not saved.
    """

    # Re-shape data for heatmap plotting
    heatmat_data = results.pivot(
        index = 'n',
        columns = 'shrink_factor',
        values = 'power'
        )
        
    # Plot
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        heatmat_data,
        annot = True,
        fmt = '.2f',
        cmap = 'viridis',
        square = True,
        linewidth = 1,
        cbar_kws = {'label': 'Power', 'shrink': 0.8, 'aspect': 10},
        ax = ax
        )
    ax.invert_yaxis()
    ax.set_xlabel('Shrink Factor')
    ax.set_ylabel('Sample Size')
    if fname:
        fig.savefig(fname, bbox_inches='tight', dpi=300)

def power_analysis(
        results_path,
        data_path,
        fig_path,
        n_per_group_levels = [50],
        shrink_factor_levels = [1.0],
        n_sims = 300,
        draws = 500,
        tune = 500,
        n_workers = 2
        ):
    """
    Perform a power analysis for a given set of parameters.

    Arguments
    ----------
    results_path : str
        Path to results directory.
    data_path : str
        Path to data directory.
    fig_path : str
        Path to figures directory.
    n_per_group_levels : list of int
        List of sample sizes per group to test.
    shrink_factor_levels : list of float
        List of shrink factors to test for the interaction effect size.
    n_sims : int
        Number of simulations to run for each combination of parameters.
    draws : int
        Number of draws for model fitting.
    tune : int
        Number of tuning steps for model fitting.
    n_workers : int
        Number of workers to parallelize simulations over.
    """

    # Generate poster draws
    trans_influence_df = load_transition_influence(
        f'{results_path}/transition-influence',
        data_path
        )
    posterior_draws = sample_from_fit_model(trans_influence_df)

    # Run power analysis for different sample sizes and shrink factors.
    results = []
    for n_per_group in n_per_group_levels:
        for shrink_factor in shrink_factor_levels:
            power = run_power_sim_posterior(
                n_per_group,
                posterior_draws,
                n_sims = n_sims,
                draws = draws,
                tune = tune,
                shrink_factor = shrink_factor,
                n_workers = n_workers
                )
            results.append([2*n_per_group, shrink_factor, power])

    # To dataframe
    results = pd.DataFrame(
        results,
        columns = ['n', 'shrink_factor', 'power']
        )
    results.to_csv(f'{results_path}/power-analysis.csv', index=False)

    # Plot
    plot_results(results, fname=f'{fig_path}/power-analysis.svg')


#######################################################################
### Run ###############################################################
#######################################################################

def main():
    
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description = 'Specify which experiment to analyse.'
        )
    parser.add_argument(
        'experiment_label',
        type = str,
        help = "Specify the file name for the config to import."
    )
    experiment_label = parser.parse_args().experiment_label
    
    # Import the experiment configuration
    analysis_config = import_config(experiment_label)

    # Run power analysis
    power_analysis(
        analysis_config['results_path'],
        analysis_config['data_path'],
        analysis_config['fig_path'],
        **analysis_config['power_analysis_config']
        )

if __name__ == '__main__':
    main()