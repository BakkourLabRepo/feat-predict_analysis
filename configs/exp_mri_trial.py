
import numpy as np

analysis_config = {

    # Data and results paths
    'data_path': '/Users/euanprentis/Documents/feat-predict-mri/exp-1/beh',
    'results_path': '/Users/euanprentis/Documents/feat-predict-mri/exp-1/derivatives/beh',

    'instance_tmat': np.array([
            [1,0,0,0],
            [1,0,0,0],
            [0,0,0,1],
            [0,0,0,1]
        ]),

    'feature_tmat': np.array([

        # Condition 1 (1I) - semantic congruent
        [[[1,0,0,0],
          [0,1,0,0],
          [0,0,1,0],
          [0,0,0,1]]],

        # Condition 2 (1II) - semantic incongruent

        [[[0,0,0,1],
          [0,0,1,0],
          [0,1,0,0],
          [1,0,0,0]]],

        # Condition 3 (2I) - semantic incongruent
        [[[0,0,1,0],
          [0,0,0,1],
          [1,0,0,0],
          [0,1,0,0]]],

        # Condition 4 (3II) - semantic congruent
        [[[1,0,0,0],
          [0,1,0,0],
          [0,0,1,0],
          [0,0,0,1]]],

        # Condition 5 (2II) - semantic congruent
        [[[1,0,0,0],
          [0,1,0,0],
          [0,0,1,0],
          [0,0,0,1]]],

        # Condition 6 (3II) - semantic incongruent
        [[[0,0,1,0],
          [0,0,0,1],
          [1,0,0,0],
          [0,1,0,0]]]

    ]),

    # Model type to fit:
    # 'agent' - per agent across trials
    # 'trial' - per trial across agents
    'model_type': 'trial',

    # Overwrite existing results?
    'overwrite': True,

    # Predict test choices based on training observations?
    'run_test_analysis': False,

    # Save trial-wise residuals? 
    'save_residuals': False,

    # Select specific subgroups to run analysis within
    'group_labels': [],

    # Number of Bambi cores to use
    'n_cores': 4,

    # Is data in BIDS format?
    'bids': True

}
