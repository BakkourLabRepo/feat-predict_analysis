
import numpy as np

PROJECT_PATH = (
    '/Users/euanprentis/Library/CloudStorage/Box-Box/Bakkour-Lab'
    '/projects/feat-predict/human/exp_1'
    )

analysis_config = {

    # Data and results paths
    'data_path': f'{PROJECT_PATH}/data',
    'results_path': f'{PROJECT_PATH}/results',
    'fig_path': f'{PROJECT_PATH}/figs',

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
    'n_cores': 4,

    # Arguments for power analysis
    'power_analysis_config': {
        'n_per_group_levels': [50, 60, 70, 80, 90, 100],
        'shrink_factor_levels': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        'n_sims': 300,
        'draws': 500,
        'tune': 500
    }

}
