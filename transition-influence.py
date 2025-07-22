import pymc as pm
import numpy as np
import pandas as pd
import argparse
import ast
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from os import listdir, makedirs



PROJECT_PATH = ''
DATA_PATH = f'{PROJECT_PATH}/data'
RESULTS_PATH = f'{PROJECT_PATH}/results'
SUCCESSOR_FEATURE_RECODING = [
    [0,1,2,3],
    [2,3,0,1]
]
OVERWRITE = True
RUN_TEST_ANALYSIS = True
SAVE_RESIDUALS = False

# Variable to group data further by within group_label
# If model_type == 'trial', var should be associated with the trial ID
# If model_type == 'agent', var should be associated with the agent ID
GROUPING_VAR = False

# Select specific project subgroups to run
GROUP_LABELS = [
]

# Number of Bambi cores to use
N_CORES = 4

def get_worker_assignments(n_cores, n_chains=4):
    """
    Get number of workers and Bambi cores, assuming 4 MCMC chains

    Arguments
    ---------
    n_cores : int
        Number of cores to parellelize over
    n_chains : int
        Number of MCMC chains
    
    Returns
    -------
    n_workers : int
        Number of workers
    n_cores_pymc : int
        Number of cores for pymc model fit
    """
    parellalize_pymc = not bool(n_cores % n_chains)
    if parellalize_pymc:
        n_workers = n_cores // n_chains
        n_cores_pymc = n_chains
    else:
        n_workers = n_cores
        n_cores_pymc = 1
    return n_workers, n_cores_pymc

def convert_str_to_array(array_string):
    """
    Convert a string representation of an array to a numpy array

    Arguments
    ---------
    array_string : str
        String representation of an array
    
    Returns
    -------
    arr : numpy.Array
        Numpy array of string
    """
    array_string = array_string.replace('\n', '')
    array_string = array_string.replace('[ ', '[')
    array_string = array_string.replace('  ', ',')
    array_string = array_string.replace(' ', ',')
    arr = ast.literal_eval(array_string)
    arr = np.array(arr)
    return arr

def get_numeric(string):
    """
    Get numeric values from a string

    Arguments
    ---------
    string : str
        String to extract numeric values from

    Returns
    -------
    string : str
        String of numeric values
    """
    string = re.findall(r'\d+', string)
    string = ''.join(string)
    return string

def unpack_feature_arrays(state_str, feature_recoding_index=False):
    """
    Upack a 1d state string into a 2d array of features (one feature
    per row)
    
    Arguments
    ---------
    state_str : str
        State string
    feature_recoding_index : list
        List of indices to recode features
        
    Returns
    -------
    f_arr : numpy.Array
        Array of integers
    """
    f_arr = convert_str_to_array(state_str)
    if feature_recoding_index:
        f_arr = f_arr[feature_recoding_index]
    f_arr = f_arr*np.eye(len(f_arr), dtype=int)
    f_arr = f_arr[np.any(f_arr, axis=1)]
    return f_arr


def sort_rows_sequentially(arr):
    """
    Sort rows of a 2d array sequentially by each column

    Arguments
    ---------
    arr : numpy.Array
        Array to sort
    
    Returns
    -------
    arr : numpy.Array
        Sorted array
    """

    # Get the number of columns
    num_cols = arr.shape[1]
    # Start with the indices of the original array
    indices = np.arange(arr.shape[0])
    
    # Sort indices based on each column sequentially
    for col in range(num_cols):
        # Get the sort order for the current column
        sort_order = np.argsort(arr[indices, col])
        # Apply this order to the indices
        indices = indices[sort_order]
    
    # Return the sorted array
    return arr[indices]


def get_possible_compositions(options, unpack_features=False):
    """
    Get all possible compositions of features given a set of options

    Arguments
    ---------
    options : list
        List of feature arrays
    unpack_features : bool
        Whether to unpack features into a single array
    
    Returns
    -------
    actions : numpy.Array
        Array of possible compositions
    """

    # Generate indices for all possible combinations of options
    n_features = len(options)
    n_per_feature = len(options[0])
    combs = np.meshgrid(*[list(range(n_per_feature))]*n_features)
    combs = np.array(combs).T.reshape(-1, n_features)
    
    # Construct set of possible compositions
    actions = []
    for comb in combs:
        if unpack_features:
            state = []
        else:
            state = np.zeros(len(options[0][0]))
        for f, i in enumerate(comb):
            if unpack_features:
                state.append(options[f][i])
            else:
                state += options[f][i]
        state = sort_rows_sequentially(np.array(state))
        actions.append(state)
    actions = np.array(actions, dtype=int)

    return actions


def recode_for_task_based_features(conj, feature_tmat):
    """
    Recode a feature array so that features are defined according to
    true transitions. I.e., if the feature transition matrix is not
    the identity matrix (between-feature transitions condition), with
    the re-defined feature array, the feature transition matrix will
    now be the identity matrix

    Arguments
    ---------
    conj : numpy.Array
        State array to recode
    feature_tmat : numpy.Array
        Feature-category transition matrix
    
    Returns
    -------
    conj : numpy.Array
        Recoded feature array
    """
    feat = np.where(conj > 0)[0][0]
    inst = conj[conj > 0][0]
    conj = inst*feature_tmat[feat]
    return conj


def get_trial_wise_target_predictions(df, successor_feature_recoding):
    """
    Get trial-wise target predictions for each possible composition

    Arguments
    ---------
    df : pandas.DataFrame
        Data frame of training data
    successor_feature_recoding : list
        List of indices to recode successor features (based on
        between-feature transition structure by condition)
    
    Returns
    -------
    predictions_df : pandas.DataFrame
        Data frame of trial-wise target predictions
    composition_successor_counts : dict
        Dictionary of composition-successor transition rates
    """

    # Fixed different labeling for simulations vs human data
    if 'trial' in df.columns:
        df['t'] = df['trial']
    if 'valid_response' in df.columns:
        df = df.loc[df['valid_response'] == 1]
        df = df.reset_index(drop=True)
            
    # Get trial-wise target predictions for each possible composition
    composition_counts = {}
    composition_successor_counts = {}
    comp_action_evidence_incidental = []
    comp_action_evidence_true = []
    all_actions = []
    trials = []
    last_edges = {}
    for t in range(len(df)):
        
        # Unpack composition, successor, and target features
        comp_features = unpack_feature_arrays(df.iloc[t]['composition'])
        succ_features = unpack_feature_arrays(
            df.iloc[t]['successor'],
            feature_recoding_index = successor_feature_recoding
        )
        target_features = unpack_feature_arrays(
            df.iloc[t]['target'],
            feature_recoding_index = successor_feature_recoding
        )
        
        # Get set of possible pairs of features for the composition
        options = df.iloc[t]['options']
        options = convert_str_to_array(options)
        actions = get_possible_compositions(options, unpack_features=True)
        
        # Get composition index in possible compositions set
        action = np.all(np.all(comp_features == actions, axis=2), axis=1)
        action = np.where(action)[0][0]

        # For each composition compute target prediction
        target_predictions = []
        for act_features in actions:
            
            # Predict each target feature separately for
            # [incidental, true] transitions
            pred = [0, 0]
            for target_f in target_features:
                

                ######### DO OUTSIDE OF THIS FUNCTION (ASSUME TRUE TRANSITIONS ARE WITHIN FEATURE)
                # Re-code features so that true transitions are
                # within column
                #target_f = recode_for_task_based_features(
                #    target_f,
                #    feature_tmat
                #)
                target_f_key = tuple(target_f)

                #true_found = False
                #incidental_found = False
                for act_f in act_features:
                    act_f_key = tuple(act_f)

                    # Feature not seen yet. Has no target prediction
                    if not act_f_key in last_edges.keys():
                        continue

                    # Is this a true or incidental transition?
                    true_edge = int(np.all(
                        target_f.astype(bool) == act_f.astype(bool)
                    ))

                    """
                    # Only add to count if edge has not been found for this target_f
                    if (
                        (true_found and true_edge) or
                        (incidental_found and not true_edge)
                    ):
                        continue
                    """

                    # If edge was observed recently, add to count
                    found = target_f_key in last_edges[act_f_key]
                    pred[true_edge] += found
                    """
                    if found:
                        if true_edge:
                            true_found = True
                        else:
                            incidental_found = True
                    """

            # Average over edge-level predictions for true vs incidental predictions
            target_predictions.append(np.array(pred)/len(target_features))

        # Make target predictions relative to action 1
        target_predictions = np.array(target_predictions) - target_predictions[0]
        target_predictions =  target_predictions[1:]

        # Skip trials with no target predictions (e.g. on first trial of block)
        if np.any(target_predictions):
            
            # Store incidental and true evidence for actions
            comp_action_evidence_incidental.append(target_predictions[:, 0])
            comp_action_evidence_true.append(target_predictions[:, 1])

            # Store action and trial number
            all_actions.append(action)
            trials.append(df.iloc[t]['t'])
    
        # For each feature of the executed action, count
        # transitions to the successors' features
        succ_keys = []
        for f_succ in succ_features:
            ######### DO OUTSIDE OF THIS FUNCTION (ASSUME TRUE TRANSITIONS ARE WITHIN FEATURE)
            # Re-code features so that true transitions are
            # within column
            #f_succ = recode_for_task_based_features(
            #    f_succ,
            #    feature_tmat
            #)
            succ_keys.append(tuple(f_succ))
        for act_f in comp_features:

            # Update last edges observed
            act_f_key = tuple(act_f)
            last_edges[act_f_key] = succ_keys

            # Add to total edge counts
            if not act_f_key in composition_counts.keys():
                composition_counts[act_f_key] = 0
                composition_successor_counts[act_f_key] = {}
            composition_counts[act_f_key] += 1
            for f_succ in succ_keys:
                if not f_succ in composition_successor_counts[act_f_key].keys():
                    composition_successor_counts[act_f_key][f_succ] = 0
                composition_successor_counts[act_f_key][f_succ] += 1

    # Add trial-wise target predictions
    comp_action_evidence_incidental = np.array(comp_action_evidence_incidental)
    comp_action_evidence_true = np.array(comp_action_evidence_true)

    # Format into data frame for saving
    predictions_df = pd.DataFrame({
        'id': df['id'].iloc[0],
        'trial': trials,
        'action': all_actions,
    })
    if 'between_cond' in df.columns:
        predictions_df['between_cond'] = df['between_cond'].iloc[0]
    for i in range(np.shape(comp_action_evidence_true)[1]):
        predictions_df[
            f'comp_action_evidence_{i + 1}_incidental'
            ] = comp_action_evidence_incidental[:, i]
        predictions_df[
            f'comp_action_evidence_{i + 1}_true'
            ] = comp_action_evidence_true[:, i]

    # Compute transition proportions
    for comp_key in composition_counts.keys():
        for succ_key in composition_successor_counts[comp_key].keys():
            composition_successor_counts[comp_key][succ_key] /= composition_counts[comp_key]

    return predictions_df, composition_successor_counts

def get_trial_wise_target_predictions_test(
        test_df,
        transition_props,
        successor_feature_recoding
    ):
    """
    Get trial-wise target predictions for each possible composition
    in the test data

    Arguments
    ---------
    test_df : pandas.DataFrame
        Data frame of test data
    transition_props : dict
        Dictionary of composition-successor training transition rates
    successor_feature_recoding : list
        List of indices to recode successor features (based on
        between-feature transition structure by condition)
    
    Returns
    -------
    predictions_df : pandas.DataFrame
        Data frame of trial-wise target predictions based on training
        transitions
    """

    # Fixed different labeling for simulations vs human data
    if 'trial' in test_df.columns:
        test_df['t'] = test_df['trial']
    if 'valid_response' in test_df.columns:
        test_df = test_df.loc[test_df['valid_response'] == 1]
        test_df = test_df.reset_index(drop=True)
            
    # Get trial-wise composition prediction
    comp_action_evidence_incidental = []
    comp_action_evidence_true = []
    all_actions = []
    trials = []
    for t in range(len(test_df)):
        
        
        # Skip trials with no valid response
        comp_features = unpack_feature_arrays(test_df.iloc[t]['composition'])
        target_features = unpack_feature_arrays(
            test_df.iloc[t]['target'],
            feature_recoding_index = successor_feature_recoding
        )

        # Get set of possible compositions
        options = test_df.iloc[t]['options']
        options = convert_str_to_array(options)
        actions = get_possible_compositions(options, unpack_features=True)
        
        # Get composition index in possible compositions set
        action = np.all(np.all(comp_features == actions, axis=2), axis=1)
        action = np.where(action)[0][0]
        
        # For each composition, compute target predictions based on
        # average training transitions
        target_predictions = []
        for act_features in actions:

            # Predict each target feature separately for
            # [incidental, true] transitions
            pred = [0, 0]
            for target_f in target_features:
                
                ######### DO OUTSIDE OF THIS FUNCTION (ASSUME TRUE TRANSITIONS ARE WITHIN FEATURE)
                # Re-code features so that "true" transitions are
                # within column
                #target_f = recode_for_task_based_features(
                #    target_f,
                #    feature_tmat
                #)
                target_f_key = tuple(target_f)

                for act_f in act_features:
                    act_f_key = tuple(act_f)

                    # Feature not seen yet
                    if not act_f_key in transition_props.keys():
                        continue
                    if not target_f_key in transition_props[act_f_key].keys():
                        continue

                    # Is this a true or incidental transition edge?
                    true_edge = int(np.all(target_f.astype(bool) == act_f.astype(bool)))
                    pred[true_edge] += transition_props[act_f_key][target_f_key]

            # Average over edge-level predictions for true vs incidental predictions
            target_predictions.append(np.array(pred)/len(target_features))

        # Make target predictions for each action relative to action 1
        target_predictions = np.array(target_predictions) - target_predictions[0]
        target_predictions =  target_predictions[1:]

        # Skip trials with no target predictions (e.g. on first trial of block)
        if np.any(target_predictions):
            
            # Store incidental and true evidence for actions
            comp_action_evidence_incidental.append(target_predictions[:, 0])
            comp_action_evidence_true.append(target_predictions[:, 1])

            # Store action and trial id
            all_actions.append(action)
            options_comb_str = get_numeric(test_df.iloc[t]['options_comb'])
            target_str = get_numeric(test_df.iloc[t]['target'])
            #target_str = ''.join([target_str[i] for i in successor_feature_recoding])
            trials.append(options_comb_str + '-' + target_str)
        
    # Add trial-wise target predictions
    comp_action_evidence_incidental = np.array(comp_action_evidence_incidental)
    comp_action_evidence_true = np.array(comp_action_evidence_true)

    # Format into data frame for saving
    predictions_df = pd.DataFrame({
        'id': test_df['id'].iloc[0],
        'trial': trials,
        'action': all_actions,
    })
    if 'between_cond' in test_df.columns:
        predictions_df['between_cond'] = test_df['between_cond'].iloc[0]
    for i in range(np.shape(comp_action_evidence_true)[1]):
        predictions_df[
            f'comp_action_evidence_{i + 1}_incidental'
            ] = comp_action_evidence_incidental[:, i]
        predictions_df[
            f'comp_action_evidence_{i + 1}_true'
            ] = comp_action_evidence_true[:, i]
        

    predictions_df = predictions_df.dropna()
    predictions_df = predictions_df.reset_index(drop=True)
    predictions_df['action'] = predictions_df['action'].astype(int)
    
    return predictions_df

def fit_transition_influence_model(
        df,
        seed = None,
        standardize = False,
        n_cores_pymc = 4
    ):
    """
    Fit a Bayesian multinomial logistic regression model to predict
    actions based on previous observations of true and incidental
    transitions from the composition to the target

    Arguments
    ---------
    df : pandas.DataFrame
        Data frame of transition predictions
    seed : int
        Random seed
    standardize : bool
        Whether to z-score predictors
    n_cores_pymc : int
        Number of cores to use for pymc model fit
    
    Returns
    -------
    trace : pymc3.backends.base.MultiTrace
        Trace of model fit
    residuals : pandas.DataFrame
        Data frame of residuals
    """
    
    
    y = df['action'].values.astype(int)

    # Number of classes and predictors
    n_actions = len(np.unique(y))

    with pm.Model() as model:

        # For each class, compute linear_combination (X@coefs_class.T)
        linear_combination = []
        for action in range(1, n_actions):

            # Class-specific variables
            cols = []
            if f'comp_action_evidence_{action}_incidental' in df.columns:
                cols.append(f'comp_action_evidence_{action}_incidental')
            if f'comp_action_evidence_{action}_true' in df.columns:
                cols.append(f'comp_action_evidence_{action}_true')
            X = df[cols].values

            # Z-score predictors
            if standardize:
                X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

            # Class-specific coefficient priors
            coefs = pm.Normal(
                f'coefs_{action}',
                mu = 0,
                sigma = 10,
                shape = (X.shape[1],)
            )

            # Linear combination of data and coefficients
            linear_combination.append(pm.math.dot(X, coefs))

        # Stack linear combinations
        linear_combination = pm.math.stack(linear_combination, axis=1)

        # Add a column of zeros for the reference class logits
        logits = pm.math.concatenate([
            np.zeros((df.shape[0], 1)),
            linear_combination
        ], axis=1)

        # Get probabilities via softmax
        probabilities = pm.math.softmax(logits, axis=1)

        # Define the likelihood (multinomial logistic regression)
        pm.Categorical('likelihood', p=probabilities, observed=y)

        # Sampling from the posterior
        n_chains = 4
        trace = pm.sample(
            2000,
            tune = 1000,
            chains = n_chains,
            cores = n_cores_pymc,
            return_inferencedata = True,
            random_seed = np.random.default_rng(seed)
        )

        # Posterior Predictive Sampling
        posterior_predictive = pm.sample_posterior_predictive(
            trace,
            var_names = ['likelihood']
        )

        # Compute predicted probabilities
        pred_action = posterior_predictive.posterior_predictive['likelihood']
        pred_action = pred_action.values
        pred_action = pred_action.reshape(n_chains, -1, len(y), 1)
        pred_action = np.eye(n_actions)[pred_action]

        # Average predictions within each chain to get predicted
        # probabilities within each chain
        predicted_probs = np.mean(pred_action, axis=1)

        # Average trial-wise probabilities across chains
        predicted_probs = np.mean(predicted_probs, axis=0)
        predicted_probs = predicted_probs.reshape(-1, n_actions)

        # Onehot encode actions
        observed_one_hot = np.eye(n_actions)[y]

        # Compute residuals
        residuals = observed_one_hot - predicted_probs

        # Format for saving
        residuals = pd.DataFrame(
            residuals,
            columns = [f'action_{i}_p' for i in range(n_actions)]
        )
        residuals['id'] = df['id'].values
        residuals['trial'] = df['trial'].values
        residuals['action'] = y
        residuals_fields = ['id', 'trial', 'action']
        for i in range(n_actions):
            residuals_fields.append(f'action_{i}_p')
        residuals = residuals[residuals_fields]
        
    return trace, residuals


def get_group_labels(data_path):
    """
    Get group labels if data subgroups exist for the project

    Arguments
    ---------
    data_path : str
        Path to data directory
    
    Returns
    -------
    group_labels : list
        List of group labels
    """
    group_labels = [
        group_label
        for group_label in listdir(data_path)
        if group_label[0] != '.'
    ]
    if 'training' in group_labels:
        group_labels = []
    return group_labels

def update_paths(data_path, results_path, group_label):
    """
    Update data and results paths with group label. If there is no
    subgroup, the original paths are returned

    Arguments
    ---------
    data_path : str
        Path to data directory
    results_path : str
        Path to results directory
    group_label : str
        Project data subgroup label
    
    Returns
    -------
    data_path : str
        Updated data path with group label
    results_path : str
        Updated results path with group label
    """
    if group_label:
        data_path = f'{DATA_PATH}/{group_label}'
        results_path = f'{RESULTS_PATH}/{group_label}'
    else:
        data_path = DATA_PATH
        results_path = RESULTS_PATH
    return data_path, results_path

def create_results_directories(
        results_path,
        model_type,
        save_residuals = False,
        run_test_analysis = False
    ):
    """
    Create results directories for saving outputs

    Arguments
    ---------
    results_path : str
        Path to results directory
    model_type : str
        Type of model to fit ('trial' or 'agent'-wise)
    save_residuals : bool
        Whether to create residuals directories
    run_test_analysis : bool
        Whether to create test directories
    """
    makedirs(
        f'{results_path}/transition-predictions/training/agent',
        exist_ok = True
    )
    makedirs(
        f'{results_path}/transition-influence/training/agent',
        exist_ok = True
    )
    if save_residuals:
        makedirs(
            f'{results_path}/transition-influence-residuals/training/agent',
            exist_ok = True
        )
    if not run_test_analysis and (model_type == 'trial'):
        phase = 'training'
        model_labels = ['trial']
    if run_test_analysis and (model_type == 'agent'):
        phase = 'test'
        model_labels = ['agent']
    elif run_test_analysis and (model_type == 'trial'):
        phase = 'test'
        model_labels = ['agent', 'trial']
    else:
        return
    for model_label in model_labels:
        makedirs(
            f'{results_path}/transition-predictions/{phase}/{model_label}',
            exist_ok = True
        )
        makedirs(
            f'{results_path}/transition-influence/{phase}/{model_label}',
            exist_ok = True
        )
        if save_residuals:
            makedirs(
                f'{results_path}/transition-influence-residuals/{phase}/{model_label}',
                exist_ok = True
            )
   

def id_from_fname(fname):
    """
    Get the agent ID from a file name

    Arguments
    ---------
    fname : str
        File name
    
    Returns
    -------
    agent_id : str
        Agent ID
    """
    fname_id = fname.split('.')[0].split('_')[1]
    if '-' in fname_id:
        fname_id = fname_id.split('-')[1]
    return fname_id

def all_ids_from_fnames(dpath):
    """
    Get all agent IDs from file names in a directory

    Arguments
    ---------
    dpath : list
        Path to directory
    
    Returns
    -------
    agent_ids : list
        List of agent IDs
    """
    return [id_from_fname(f) for f in listdir(dpath) if f[0] != '.']

def get_files_to_analyse(
        input_path,
        output_path,
        overwrite = False
    ):
    """
    Get files to analyse based on whether results already exist

    Arguments
    ---------
    input_path : str
        Path to input data
    output_path : str
        Path to output results
    overwrite : bool
        Whether to overwrite existing results
    
    Returns
    -------
    fnames : list
        File names for files to analyse
    """

    if overwrite:
        return listdir(input_path)

    # Get agent IDs for input data and existing results
    data_ids = all_ids_from_fnames(input_path)
    results_ids = all_ids_from_fnames(output_path)

    # Get IDs not yet analysed
    ids_to_run = list(set(data_ids).difference(set(results_ids)))

    # Select files to analyse
    fnames = [
        f for f in listdir(input_path)
        if (id_from_fname(f) in ids_to_run) and f[0] != '.'
    ]

    return fnames

def compute_transition_predictions(
        training_path = None,
        test_path = False,
        results_path = ''
    ):
    """
    Compute transition predictions for a given training and test set

    Arguments
    ---------
    training_path : str
        Path to training data
    test_path : str
        Path to test data. If False, test analysis is not run
    results_path : str
        Path to parent results directory
    
    Returns
    -------
    None
    """

    # Load training data
    training_df = pd.read_csv(training_path)

    # Get number of environment features
    target_comb = training_df['target_comb'].values[0]
    target_comb = convert_str_to_array(target_comb)

    # Get between-feature transition matrix
    ###### UPDATE THIS TO BE FLEXIBLE WITH MRI CONDITIONS
    #feature_tmat = np.eye(n_feats, dtype=int)
    if 'between_cond' in training_df.columns:
        condition = training_df['between_cond'].values[0]
        successor_feature_recoding = SUCCESSOR_FEATURE_RECODING[condition]
    #    if training_df['between_cond'].values[0]:
    #        feature_tmat = feature_tmat[FEATURE_RECODING_INDEX]
            
    # Get training trial-wise target predictions and phase-wise
    # transition proportions
    transitions_df, transition_props = get_trial_wise_target_predictions(
        training_df,
        successor_feature_recoding
    )

    # Save the training transition predictions
    agent_id = id_from_fname(training_path.split('/')[-1])
    output_fname = f'transition-predictions_agent-{agent_id}.csv'
    transitions_df.to_csv(
        f'{results_path}/transition-predictions/training/agent/{output_fname}',
        index = False
    )

    # Compute test transition predictions absed on training transitions
    if test_path:

        # Load test data
        test_df = pd.read_csv(test_path)

        # Get test trial-wise target predictions
        test_transitions_df = get_trial_wise_target_predictions_test(
            test_df,
            transition_props,
            successor_feature_recoding
        )

        # Save the test transition predictions
        test_transitions_df.to_csv(
            f'{results_path}/transition-predictions/test/agent/{output_fname}',
            index = False
        )

def save_trial_wise_transition_predictions(results_path):
    """
    Save all transition predictions for each trial in one file

    Arguments
    ---------
    results_path : str
        Path to agent-wise transition predictions results directory

    Returns
    -------
    None
    """

    # Load all transition predictions into one file
    pred_df = pd.concat([
        pd.read_csv(f'{results_path}/{f}', index_col=False)
        for f in listdir(results_path)
        ])
    pred_df = pred_df.reset_index(drop=True)

    # Save transition predictions seperately for each trial
    trial_ids = np.unique(pred_df['trial'])
    results_path = results_path.replace('agent', 'trial')
    for trial in trial_ids:
        trial_idx = pred_df['trial'] == trial

        # Split data by condition, if applicable
        trial_dfs = {}
        if 'between_cond' in pred_df.columns:
            for condition in np.unique(pred_df['between_cond']):
                trial_dfs[f'_condition-{condition}'] = pred_df[
                        (trial_idx) &
                        (pred_df['between_cond'] == condition)
                    ]
        else:
            trial_dfs[''] = pred_df[trial_idx]
        
        # Save data if there are predictions
        for label, trial_df in trial_dfs.items():

            # Drop empty data frames
            if trial_df.empty:
                continue

            # Drop columns with no variance
            cols_to_drop = []
            for col in trial_df.columns:
                if (
                    ('evidence' in col) and
                    (len(np.unique(trial_df[col])) == 1)
                    ):
                    cols_to_drop.append(col)
            trial_df = trial_df.drop(cols_to_drop, axis=1)

            fname = f'transition-predictions_trial-{trial}{label}.csv'
            trial_df.to_csv(f'{results_path}/{fname}', index=False)

def run_transition_influence_analysis(
    input_path,
    output_path,
    residuals_path,
    n_cores_pymc = 4,
    seed = None,
    standardize = False
):
    """
    Run transition influence analysis

    Arguments
    ---------
    input_path : str
        Path to transition predictions file
    output_path : str
        Path to save model fit results file
    residuals_path : str
        Path to save residuals file
    n_cores_pymc : int
        Number of cores to use for pymc model fit
    seed : int
        Random seed
    standardize : bool
        Whether to z-score predictors
    
    Returns
    -------
    None
    """

    print(f'Fitting model for {input_path}...')

    # Load data
    transitions_df = pd.read_csv(input_path)

    # Fit model
    trace, residuals = fit_transition_influence_model(
        transitions_df,
        seed = seed,
        standardize = standardize,
        n_cores_pymc = n_cores_pymc
    )

    # Save results
    with open(output_path, 'wb') as f_out:
        pickle.dump(trace, f_out)

    # Save residuals
    if residuals_path:
        residuals.to_csv(residuals_path, index=False)


def main():

    ###################################################################
    ###  Get command lind arguments ###################################
    ###################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_type',
        type = str,
        help = 'Fit model per trial ("trial") or per agent ("agent").'
    )
    model_type = parser.parse_args().model_type

    # Cannot run test analysis for trial-wise models
    #if model_type == 'trial':
    #    RUN_TEST_ANALYSIS = False

    ###################################################################
    ###  Setup configurations for analyses ############################
    ###################################################################

    # Get number of workers and cores for pymc
    n_workers, n_cores_pymc = get_worker_assignments(N_CORES)

    # Get subgroup labels
    if not GROUP_LABELS:
        group_labels = get_group_labels(DATA_PATH)
    else:
        group_labels = GROUP_LABELS

    # Account for projects with no group labels
    if not group_labels:
        group_labels = [False]
    
    # Init lists to store transition prediction configurations
    trans_pred_configs = []
    trialwise_trans_pred_configs = []
    
    # Run analyses seperately for each group
    for group_label in group_labels:

        # Update data and results paths with group label
        data_path, results_path = update_paths(
            DATA_PATH,
            RESULTS_PATH,
            group_label
        )

        # Create results directories
        create_results_directories(
            results_path,
            model_type,
            save_residuals = SAVE_RESIDUALS,
            run_test_analysis = RUN_TEST_ANALYSIS
        )

        # Get training files to generate transition predictions for
        if RUN_TEST_ANALYSIS:
            phase = 'test'
        else:
            phase = 'training'
        fnames = get_files_to_analyse(
            f'{data_path}/{phase}',
            f'{results_path}/transition-predictions/{phase}/agent',
            overwrite = OVERWRITE
        )

        # Format agent-wise transition prediction configurations
        for fname in fnames:
            if RUN_TEST_ANALYSIS:
                training_fname = fname.replace('test', 'training')
                training_path = f'{data_path}/training/{training_fname}'
                test_path = f'{data_path}/test/{fname}'
            else:
                training_path = f'{data_path}/training/{fname}'
                test_path = False
            trans_pred_configs.append({
                'training_path': training_path,
                'test_path': test_path,
                'results_path': results_path
            })

        # Format trial-wise transition prediction configurations
        if model_type == 'trial':
            trialwise_trans_pred_configs.append(
                f'{results_path}/transition-predictions/{phase}/agent'
            )


    ###################################################################
    ### Compute agent-wise transition predictions #####################
    ###################################################################
    print('Computing transition predictions...')

    # Parellelize transition prediction computations
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        [
            executor.submit(
                compute_transition_predictions,
                **trans_pred_config
            )
            for trans_pred_config in trans_pred_configs
        ]


    ###################################################################
    ### Re-format transition predictions for trial-wise analysis ######
    ###################################################################

    if model_type == 'trial':
        print('Formatting trial-wise transition predictions...')

        # Parellelize re-formatting of transition predictions
        with ProcessPoolExecutor(max_workers=N_CORES) as executor:
            [
                executor.submit(
                    save_trial_wise_transition_predictions,
                    results_path
                )
                for results_path in trialwise_trans_pred_configs
            ]


    ###################################################################
    ### Get model fitting configurations ##############################
    ###################################################################

    # Init model fit configurations
    model_fit_configs = []

    # Get phases to run analyses for
    if RUN_TEST_ANALYSIS:
        phases = ['training', 'test']
        phases = ['test']
    else:
        phases = ['training']

    # Run analyses seperately for each group
    for group_label in group_labels:

        # Update data and results paths with group label
        results_path = update_paths(DATA_PATH, RESULTS_PATH, group_label)[1]

        for phase in phases:

            # Get training transition prediction files to fit models to
            fnames = get_files_to_analyse(
                f'{results_path}/transition-predictions/{phase}/{model_type}',
                f'{results_path}/transition-influence/{phase}/{model_type}',
                overwrite = OVERWRITE
            )
            for fname in fnames:

                # Output name for pymc trace
                output_fname = fname.replace('predictions', 'influence')
                output_fname = output_fname.replace('.csv', '.pkl')

                # Residuals path
                if SAVE_RESIDUALS:
                    residuals_fname = fname.replace('predictions', 'influence-residuals')
                    residuals_path = f'{results_path}/transition-influence-residuals/{phase}/{model_type}/{residuals_fname}'
                else:
                    residuals_path = False

                # Only standardize at test
                if phase == 'test':
                    standardize = True
                else:  
                    standardize = False
                
                # Format model fit configuration
                model_fit_configs.append({
                    'input_path': f'{results_path}/transition-predictions/{phase}/{model_type}/{fname}',
                    'output_path': f'{results_path}/transition-influence/{phase}/{model_type}/{output_fname}',
                    'residuals_path': residuals_path,
                    'n_cores_pymc': n_cores_pymc,
                    'seed': int(id_from_fname(fname)),
                    'standardize': standardize
                })
    
    ###################################################################
    ### Fit transition influence models ###############################
    ###################################################################

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        [
            executor.submit(
                run_transition_influence_analysis,
                **model_fit_config
            )
            for model_fit_config in model_fit_configs
        ]


if __name__ == '__main__':
    main()
