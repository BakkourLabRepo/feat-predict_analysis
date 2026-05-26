
import numpy as np

analysis_config = {

    # Data and results paths
    'data_path': '/Users/euanprentis/Library/CloudStorage/Box-Box/Bakkour-Lab/projects/feat-predict/human/exp_1/data',
    'results_path': '/Users/euanprentis/Library/CloudStorage/Box-Box/Bakkour-Lab/projects/feat-predict/human/exp_1/results_2',

    'instance_tmat': np.array([
            [1,0,0,0],
            [1,0,0,0],
            [0,0,0,1],
            [0,0,0,1]
        ]),

    'feature_tmat': [

        # Semantic congruent
        [

            np.array([
                [1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]
            ])

        ],

        # Semantic incongruent
        [

            np.array([
                [0,0,1,0],
                [0,0,0,1],
                [1,0,0,0],
                [0,1,0,0]
            ])

        ],

    ],

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
    'n_cores': 4

}
