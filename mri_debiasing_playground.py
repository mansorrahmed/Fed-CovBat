import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os, glob, re
from neurocombat_sklearn import CombatModel
from baselines.ComBat.neuroCombat import neuroCombat
from name_that_site import ClassificationModelTrainer
from brain_age_estimation import RegressionModelTrainer
import patsy
import time
import sys
import baselines.CovBat.covbat as cb
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

def main():

    openbhb_dir = "/Users/mansoor/Documents/GSU/Projects/Neuroimaging/Dataset/OpenBHB/"
    roi_dir = os.path.join(openbhb_dir , "roi/")
    debiased_roi_dir = os.path.join(openbhb_dir , "debiased_roi/")
    labels_dir = os.path.join(openbhb_dir , "labels/")
    results_dir = "results/"
    results_name_that_dataset_dir = os.path.join(results_dir, "name_that_dataset/")
    results_brain_age_estimation_dir = os.path.join(results_dir, "brain_age_estimation/")

    f_participants = os.path.join(labels_dir, "participants.csv")
    f_sites = os.path.join(labels_dir, "site_labels.csv")

    roi_files = glob.glob(os.path.join(roi_dir, "*.csv"))

    participants = pd.read_csv(f_participants)
    sites = pd.read_csv(f_sites, index_col=False)
    features_set = ["desikan_roi", "desikan_cortical_thickness", "normalized_vbm_roi", "destrieux_intrinsic_curvature_index",
        "destrieux_gray_matter_volume", "desikan_gray_matter_volume", "desikan_integrated_rectified_gaussian_curvature",
        "destrieux_surface_area", "desikan_average_thickness", "desikan_surface_area", "destrieux_thickness_stddev",
        "desikan_intrinsic_curvature_index", "destrieux_integrated_rectified_mean_curvature", "destrieux_roi",
        "desikan_integrated_rectified_mean_curvature", "destrieux_integrated_rectified_gaussian_curvature", "desikan_thickness_stddev"]

    # loop through the mri features sets, apply harmonization, and perform downstreams classification and regression tasks
    for mri_features_file, feature_set_i in zip(roi_files, features_set):

        desikan_thickness = pd.read_csv(mri_features_file)
        desikan_thickness_df = desikan_thickness.merge(sites.loc[:, ['participant_id', 'siteXacq']], on="participant_id")
        desikan_thickness_df = desikan_thickness_df.merge(participants.loc[:,["participant_id", "sex", "age"]])
        covariates = desikan_thickness_df.loc[:,["siteXacq", "sex", "age"]]

        unharmonized = np.array(desikan_thickness_df.iloc[:,1:-3]) 
        site_labels = np.array(desikan_thickness_df["siteXacq"])
        age_labels = np.array(desikan_thickness_df["age"])

        ####  Run Combat ####
        combat_harmonized = neuroCombat( data=unharmonized, covars=covariates, batch_col="siteXacq", continuous_cols=["age"])
        pd.DataFrame(combat_harmonized).to_csv(debiased_roi_dir +  f"combat/{feature_set_i}_combat_harmonized.csv") 

        #### Run CovBat ####
        model = patsy.dmatrix("~ age + sex", covariates, return_type="dataframe")
        t = time.time()
        covbat_harmonized = cb.covbat(pd.DataFrame(unharmonized).transpose(), covariates['siteXacq'].transpose(), model, "age")
        pd.DataFrame(covbat_harmonized).transpose().to_csv(debiased_roi_dir +  f"covbat/{feature_set_i}_covbat_harmonized.csv") 
        covbat_harmonized = np.array(covbat_harmonized).transpose()
        sys.stdout.write("%.2f seconds\n" % (time.time() - t))

        #### Initialize DataFrames for classification and regression (brain age estimation) Results ####
        mean_df = pd.DataFrame()
        full_df = pd.DataFrame()
        mean_std_df = pd.DataFrame()

        mean_df_reg = pd.DataFrame()
        full_df_reg = pd.DataFrame()
        mean_std_df_reg = pd.DataFrame()

        #### Train classifiers for each harmonization strategy ####
        harmonization_strategies = ['unharmonized', 'combat', 'covbat']
        datasets = [unharmonized, combat_harmonized, covbat_harmonized]
        print(f"Started training/testing models on {feature_set_i} dataset..")

        for strategy, dataset in zip(harmonization_strategies, datasets):
            stratify_cols = covariates[['age', 'sex', 'siteXacq']].apply(tuple, axis=1).factorize()[0]

            # Run with stratified split
            classifiers = ClassificationModelTrainer(X=dataset, y=site_labels, results_dir=results_name_that_dataset_dir,
                                    harmonization_strategy=strategy, dataset=feature_set_i, stratify_cols=stratify_cols)
            mean_df, full_df, mean_std_df = classifiers.run(mean_df, full_df, mean_std_df, stratify=True)

            # Run with random split
            classifiers = ClassificationModelTrainer(X=dataset, y=site_labels, results_dir=results_name_that_dataset_dir,
                                    harmonization_strategy=strategy, dataset=feature_set_i)
            mean_df, full_df, mean_std_df = classifiers.run(mean_df, full_df, mean_std_df, stratify=False)

                # Run with stratified split
            regressors = RegressionModelTrainer(X=dataset, y=age_labels, results_dir=results_brain_age_estimation_dir,
                                                harmonization_strategy=strategy, dataset=feature_set_i,
                                                stratify_cols=stratify_cols)
            mean_df_reg, full_df_reg, mean_std_df_reg = regressors.run(mean_df_reg, full_df_reg, mean_std_df_reg, stratify=True)

            # Run with random split
            regressors = RegressionModelTrainer(X=dataset, y=age_labels, results_dir=results_brain_age_estimation_dir,
                                                harmonization_strategy=strategy, dataset=feature_set_i)
            mean_df_reg, full_df_reg, mean_std_df_reg = regressors.run(mean_df_reg, full_df_reg, mean_std_df_reg, stratify=False)

        #### Save the accumulated brain age estimation and "name that site" results ####
        mean_df_reg.to_csv(os.path.join(results_brain_age_estimation_dir, f'brain_age_estimation_mean_results_{feature_set_i}.csv'), index=True)
        full_df_reg.to_csv(os.path.join(results_brain_age_estimation_dir, f'brain_age_estimation_full_k_folds_results_{feature_set_i}.csv'), index=True)
        mean_std_df_reg.to_csv(os.path.join(results_brain_age_estimation_dir, f'brain_age_estimation_mean_std_results_{feature_set_i}.csv'), index=True)

        mean_df.iloc[:,1:].to_csv(os.path.join(results_name_that_dataset_dir, f'name_that_site_mean_results_{feature_set_i}.csv'), index=True)
        full_df.to_csv(os.path.join(results_name_that_dataset_dir, f'name_that_site_full_k_folds_results_{feature_set_i}.csv'), index=True)
        mean_std_df.iloc[:,1:].to_csv(os.path.join(results_name_that_dataset_dir, f'name_that_site_mean_std_results_{feature_set_i}.csv'), index=True)
        print(f"Complete results saved for {feature_set_i} dataset..")


    # # =========================
    # # Step 1: Parse Command-Line Arguments
    # # =========================
    # parser = argparse.ArgumentParser(description='Data Preprocessing and Model Training Script')
    # parser.add_argument(
    #     '--proj_dir', 
    #     type=str,
    #     required=True
    #     )
    # parser.add_argument(
    #     '--strategy', 
    #     type=str,
    #     required=False,
    #     choices=["mean_imp", "median_imp", "ffill_bfill_imp", "linear_interp_imp", "kalman_imp", "mice_imp", "knn_imp", "lstm_imp", "rnn_imp"],
    #     help='Imputation strategy to use. Choices are: mean_imp, median_imp, ffill_bfill_imp, linear_interp_imp, kalman_imp, mice_imp, knn_imp, lstm_imp, rnn_imp'
    # )
    # parser.add_argument(
    #     '--epochs', 
    #     type=int,
    #     required=False
    #     )
    # parser.add_argument(
    #     '--models', 
    #     type=str,
    #     required=False
    #     )
    # args = parser.parse_args()
    # strategy = args.strategy
    # proj_dir = args.proj_dir
    # epochs = args.epochs
    # models_category = args.models

   
    
if __name__ == "__main__":
    main()
