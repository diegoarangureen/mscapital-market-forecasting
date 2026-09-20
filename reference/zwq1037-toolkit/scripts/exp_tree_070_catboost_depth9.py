"""Run the 307-feature GPU CatBoost control with depth increased from 7 to 9."""

import exp_tree_069_catboost_exp053r_features as experiment


BASE_PARAMETERS = experiment.model_parameters


def depth9_parameters(iterations: int) -> dict:
    parameters = BASE_PARAMETERS(iterations)
    parameters["depth"] = 9
    return parameters


if __name__ == "__main__":
    experiment.EXPERIMENT_ID = "EXP-TREE-070"
    experiment.model_parameters = depth9_parameters
    experiment.main()
