import requests
import json
import pandas as pd

if __name__ == "__main__":

    player_url = "https://fantasy.premierleague.com/drf/element-summary/69"
    print(player_url)
    r = requests.get(player_url)

    # skip non-existent players
    if r.status_code != 200:
        api_error = r.status_code
        print('Not found.')
        print('\n')

    # assign player name to dict key, write all json data to key
    temp = r.json()

    #print(temp["history"])
    with open('69.json', 'w') as outfile:
        #json.dump(temp, outfile)
        json.dump(temp["history"],outfile)

    df = pd.DataFrame(temp["history"])


    print(df.head(100))
    print(df.id)

