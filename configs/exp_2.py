
import numpy as np

analysis_config = {

    # Data and results paths
    'data_path': '/Users/euanprentis/Library/CloudStorage/Box-Box/Bakkour-Lab/projects/feat-predict/human/exp_2/data',
    'results_path': '/Users/euanprentis/Library/CloudStorage/Box-Box/Bakkour-Lab/projects/feat-predict/human/exp_2/results',

    'instance_tmat': np.array([
            [1,0,0,0,0,0],
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
            [0,0,0,0,0,1]
        ]),

    'feature_tmat': np.array([

        # Condition 1 (1I) - semantic congruent
        [

            # Step 1
            [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]],

            # Step 2
            [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]],

        ],

        # Condition 2 (1II) - semantic incongruent

        [
            # Step 1
            [[0,0,0,1],
             [0,0,1,0],
             [0,1,0,0],
             [1,0,0,0]],

            # Step 2
            [[0,0,1,0],
             [0,0,0,1],
             [1,0,0,0],
             [0,1,0,0]],

        ],

        # Condition 3 (2I) - semantic incongruent
        [

            # Step 1
            [[0,0,1,0],
             [0,0,0,1],
             [1,0,0,0],
             [0,1,0,0]],

            # Step 2
            [[0,0,0,1],
             [0,0,1,0],
             [0,1,0,0],
             [1,0,0,0]],

        ],

        # Condition 4 (3II) - semantic congruent
        [

            # Step 1
            [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]],

            # Step 2
            [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]],

        ],

        # Condition 5 (2II) - semantic congruent
        [

            # Step 1
            [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]],

            # Step 2
            [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]],

        ],

        # Condition 6 (3II) - semantic incongruent
        [
            # Step 1
            [[0,0,1,0],
             [0,0,0,1],
             [1,0,0,0],
             [0,1,0,0]],

            # Step 2
            [[0,0,0,1],
             [0,0,1,0],
             [0,1,0,0],
             [1,0,0,0]]

        ]
        
    ]),

    # Model type to fit:
    # 'agent' - per agent across trials
    # 'trial' - per trial across agents
    'model_type': 'agent',

    # Overwrite existing results?
    'overwrite': True,

    # Predict test choices based on training observations?
    'run_test_analysis': True,

    # Save trial-wise residuals? 
    'save_residuals': False,

    # Select specific subgroups to run analysis within
    'group_labels': [],

    # Number of Bambi cores to use
    'n_cores': 4

}
