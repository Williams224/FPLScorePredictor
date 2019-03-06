import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


if __name__ == "__main__":

    df_training = pd.read_csv("Data/training_set_features/2019-03-02/training_set_featurised.csv")

    df_training = df_training.drop("Unnamed: 0", axis = 1)

    n_full_training_set = len(df_training)

    min_minutes_per_game = 20

    df_training = df_training.assign(meets_min_minutes =
                                     df_training["total_minutes_played"] > min_minutes_per_game*df_training["round"])

    filtered_training_set = df_training[df_training["meets_min_minutes"] == True]

    filtered_training_set = filtered_training_set[filtered_training_set["played_next_game"] == True]

    filtered_training_set = filtered_training_set[filtered_training_set["played_last_3_games"] == True]

    correlations = filtered_training_set.corr()

    target_correlations = correlations["target"]

    sorted_target_correlations = target_correlations.sort_values(ascending=False)

    top_ten = sorted_target_correlations.values[1:11]

    top_ten_names = sorted_target_correlations.index[1:11]

    top_ten_names_list = top_ten_names.values.tolist()

    top_ten_names_list.append("target")

    top_ten_df = filtered_training_set[top_ten_names_list]

    pp = sns.pairplot(top_ten_df, diag_kind='kde')

    pp.savefig("top_ten_pairplot.pdf")

    favoured_features = ["value", "threat_per_minute", "ict_index_per_minute",
                        "total_points_per_minute","goals_scored_per_minute",
                        "selected","big_chances_missed_per_minute","creativity_per_minute"
                        ,"offside_per_minute","winning_goals_per_minute"]

    favoured_features.append("target")

    pp_favoured_features_df = filtered_training_set[favoured_features]

    pp_favoured = sns.pairplot(pp_favoured_features_df, diag_kind='kde')

    pp_favoured.savefig("favoured_features_pairplot.pdf")

    print("Done")

