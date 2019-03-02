import requests
import json
import pandas as pd
import time

def get_static_data():
    static_url = "https://fantasy.premierleague.com/drf/bootstrap-static"
    r = requests.get(static_url)
    if r.status_code != 200:
        api_error = r.status_code
        print('Not found.')
        print('\n')

        # assign player name to dict key, write all json data to key
    temp = r.json()

    return temp

def get_player_data_raw(player_id):
    player_url = "https://fantasy.premierleague.com/drf/element-summary/{pid}".format(pid=player_id)
    r = requests.get(player_url)
    if r.status_code != 200:
        api_error = r.status_code
        print('Not found.')
        print('\n')

        # assign player name to dict key, write all json data to key
    temp = r.json()

    return temp

def create_player_history_table(player_id, player_json, df_static):
    player_history = pd.DataFrame(player_json["history"])

    player_history = player_history.assign(player_id = player_id)

    player_history = player_history.rename(columns = {"id":"performance_id"})

    player_history_with_names = player_history.merge(df_static[["id","first_name","second_name"]],
                                                     how = 'left', left_on ="player_id", right_on = "id")

    return player_history_with_names

def create_raw_player_json_path(date, player_id):
    return "Data/raw_elements/{date}/{pid}.json".format(date=date, pid=player_id)

def create_player_data_path(date, player_id):
    return "Data/player_data/{date}/{pid}.csv".format(date=date, pid=player_id)


if __name__ == "__main__":

    static_data_json = get_static_data()

    df_static = pd.DataFrame(static_data_json["elements"])

    for player_id in range(1, 606):
        print(player_id)
        player_json = get_player_data_raw(player_id)

        with open(create_raw_player_json_path("2019-03-02", player_id),"w") as out_file:
           json.dump(player_json, out_file)

        player_history = create_player_history_table(player_id, player_json, df_static)

        player_history.to_csv(create_player_data_path("2019-03-02",player_id))

        time.sleep(0.5)






