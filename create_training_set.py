import pandas as pd
import numpy as np
import retrieve_data
import multiprocessing

def count_games_played(minutes, threshold=20):
   games_played = minutes[minutes > threshold].count()
   return games_played

def create_featurised_player_fp(date, player_id):
    return "Data/featurised_players/{date}/{pid}.csv".format(date=date, pid=player_id)

def played_last_n(minutes):
    any_not_played = (minutes > 20).all()
    return any_not_played


def featurise_player_data(date, player_id, n_game_form = 3, game_threshold = 20, full_game_threshold = 60):

    print("Featurising data for player = ", player_id)

    df_player = pd.read_csv(retrieve_data.create_player_data_path(date, player_id))

    df_player = df_player.drop(["Unnamed: 0", "id"], axis=1)

    df_player = df_player.assign(target=df_player.total_points.shift(-1))

    df_player["games_played"] = \
        df_player.expanding(min_periods=1).minutes.apply(lambda x: count_games_played(x, game_threshold), raw=False)

    df_player["total_minutes_played"] = df_player.expanding(min_periods=1).minutes.sum()

    variables = ["assists", "attempted_passes", "big_chances_created", "big_chances_missed", "bonus", "bps",
                 "clean_sheets",
                 "clearances_blocks_interceptions", "completed_passes", "creativity", "dribbles",
                 "errors_leading_to_goal",
                 "errors_leading_to_goal_attempt", "fouls", "goals_conceded", "goals_scored", "ict_index", "influence",
                 "offside",
                 "open_play_crosses", "opponent_team", "own_goals", "penalties_missed", "penalties_saved", "recoveries",
                 "red_cards", "saves",
                 "tackled", "tackles", "target_missed", "threat", "total_points", "value", "winning_goals",
                 "yellow_cards"]

    new_columns_dict = {}

    for var in variables:
        new_columns_dict["{var}_per_minute".format(var=var)] = \
            df_player[var].expanding(1).sum() / df_player["total_minutes_played"]

        new_columns_dict["{var}_per_game".format(var=var)] = df_player[var].expanding(1).sum() / df_player["round"]

        new_columns_dict["{var}_per_played_game".format(var=var)] = \
            df_player[var].expanding(1).sum() / df_player["games_played"]

        new_columns_dict["{var}_{n}_game_form".format(var=var, n=n_game_form)] = df_player[var].rolling(3).mean()

    new_columns_dict["played_last_{n}_games".format(n=n_game_form)] = df_player.minutes.rolling(3).apply(played_last_n)

    new_columns_dict["next_game_minutes"] = df_player.minutes.shift(-1)

    df_player = df_player.assign(**new_columns_dict)

    df_player = df_player.assign(played_next_game = df_player.next_game_minutes > full_game_threshold)

    featurised_file = create_featurised_player_fp(date, player_id)

    df_player.to_csv(featurised_file)

    return df_player

if __name__ == "__main__":

    data_date = "2019-03-02"

    featurised_dfs = list(map(lambda x: featurise_player_data(data_date, x), np.arange(1,606)))

    training_set = pd.concat(featurised_dfs,ignore_index=True)

    training_set.to_csv("Data/training_set_features/2019-03-02/training_set_featurised.csv")

    print(training_set)




# Load a player
# Add targets
# Add week on week features
# Filter performance if avg minutes played < 45
# Require last 3 performances to have played > 60 mins



