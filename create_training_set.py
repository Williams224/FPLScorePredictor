import pandas as pd
import numpy as np
import retrieve_data

def count_games_played(minutes, threshold=20):
   games_played = minutes[minutes > threshold].count()
   return games_played


if __name__ == "__main__":

    df_player = pd.read_csv(retrieve_data.create_player_data_path("2019-03-02", 270))

    df_player = df_player.drop(["Unnamed: 0", "id"], axis=1)

    df_player = df_player.assign(target=df_player.total_points.shift(-1))

    minutes_played = df_player.minutes.sum()
    played_games = len(df_player.minutes[df_player.minutes > 20])

    df_player["games_played"] = \
        df_player.expanding(min_periods=1).minutes.apply(lambda x: count_games_played(x, 20), raw=False)

    df_player["total_minutes_played"] = df_player.expanding(min_periods=1).minutes.sum()

    variables = ["assists", "attempted_passes", "big_chances_created", "big_chances_missed", "bonus", "bps", "clean_sheets",
     "clearances_blocks_interceptions", "completed_passes", "creativity", "dribbles", "errors_leading_to_goal",
     "errors_leading_to_goal_attempt", "fouls", "goals_conceded", "goals_scored", "ict_index", "influence", "offside",
     "open_play_crosses", "opponent_team", "own_goals", "penalties_missed", "penalties_saved", "recoveries", "red_cards","saves",
    "tackled", "tackles", "target_missed", "threat", "total_points","value", "winning_goals", "yellow_cards"]

    new_columns_dict = {}

    for var in variables:
        new_columns_dict["{var}_per_minute".format(var=var)] = \
            df_player[var].expanding(1).sum()/df_player["total_minutes_played"]

        new_columns_dict["{var}_per_game".format(var=var)] = df_player[var].expanding(1).sum()/df_player["round"]

        new_columns_dict["{var}_per_played_game".format(var=var)] = \
            df_player[var].expanding(1).sum()/df_player["games_played"]

    df_player = df_player.assign(**new_columns_dict)

    

    print(df_player.head(100))



# Load a player
# Add targets
# Add week on week features
# Filter performance if avg minutes played < 45
# Require last 3 performances to have played > 60 mins



