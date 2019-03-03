import pandas as pd
import numpy as np
import retrieve_data


if __name__ == "__main__":

    df_player = pd.read_csv(retrieve_data.create_player_data_path("2019-03-02", 270))

    df_player = df_player.drop(["Unnamed: 0", "id"], axis=1)

    df_target = df_player.assign(target=df_player.total_points.shift(-1))

    minutes_played = df_target.minutes.sum()
    played_games = len(df_target.minutes[df_target.minutes > 20 ])



    print(df.head(100))



# Load a player
# Add targets
# Add week on week features
# Filter performance if avg minutes played < 45
# Require last 3 performances to have played > 60 mins



