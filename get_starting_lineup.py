from collections import Counter
from datetime import datetime
import numpy as np
import os
from turtle import end_poly
import pandas as pd
import pybaseball
from pybaseball import playerid_reverse_lookup, playerid_lookup
import statsapi
from pprint import pprint
from tqdm import tqdm
from tqdm.contrib.itertools import product

DEBUG = False

# team name to teamid
TEAMS_TO_CODE = {
    'BAL': 110, 'BOS': 111, 'NYY': 147, 'TBR': 139, 'TOR': 141,
    'CHW': 145, 'CLE': 114, 'DET': 116, 'KCR': 118, 'MIN': 142,
    'HOU': 117, 'LAA': 108, 'OAK': 133, 'SEA': 136, 'TEX': 140,
    'ATL': 144, 'MIA': 146, 'NYM': 121, 'PHI': 143, 'WSN': 120,
    'CHC': 112, 'CIN': 113, 'MIL': 158, 'PIT': 134, 'STL': 138,
    'ARI': 109, 'COL': 115, 'LAD': 119, 'SDP': 135, 'SFG': 137,
}

CODE_TO_TEAM = {v: k for k, v in TEAMS_TO_CODE.items()}

YEAR_TEAM_GAMES = {
    year : {team : 162 if year != 2020 else 60 for team in TEAMS_TO_CODE.keys()} for year in range(2015, 2022+1)
}

YEAR_TEAM_NGAMES_EXCEPTION = {
    2022 : {},
    2021 : {
        'COL': 161, 'ATL': 161
    },
    2020 : {
        'DET': 58, 'STL': 58
    },
    2019 : { 
        'CHW': 161, 'DET': 161
    },
    2018 : {
        'MIA' : 161, 'PIT': 161, 'MIL': 163, 'CHC': 163, 'LAD': 163, 'COL': 163
    },
    2017 : {},
    2016 : {
        'CLE' : 161, 'DET' : 161, 'MIA' : 161, 'ATL' : 161, 'CHC' : 161, 'PIT' : 161
    },
    2015 : {
        'CLE' : 161, 'DET' : 161
    }
}

for year in range(2015, 2022+1):
    YEAR_TEAM_GAMES[year].update(YEAR_TEAM_NGAMES_EXCEPTION[year])

def get_game_pk(team, year):
    games = [
        game for game in statsapi.schedule(
            start_date=f'01/01/{year}', end_date=f'12/31/{year}', team=TEAMS_TO_CODE[team]
        ) 
        if game['game_type']=='R' and (game['status'] != 'Cancelled' and game['status'] != 'Postponed')
    ]
    # 날짜, 게임키, 승패여부 가져오기
    games_df = pd.DataFrame(games)
    n_gmkey = len(games_df.game_id.unique())
    home_newline = sum(games_df.home_pitcher_note.str.contains('\n'))
    away_newline = sum(games_df.away_pitcher_note.str.contains('\n')) # \n 삭제하면 될듯? 아니면 ' '로 대체 (얘가 안전해 보임)
    if home_newline != 0 or away_newline != 0:
        # replace '\n' with ' '
        games_df.home_pitcher_note = games_df.home_pitcher_note.str.replace('\n', ' ')
        games_df.away_pitcher_note = games_df.away_pitcher_note.str.replace('\n', ' ')
    
    try:
        assert n_gmkey == YEAR_TEAM_GAMES[year][team]
    except AssertionError as e:
        print(f'AssertionError: {team} {year} {n_gmkey} {YEAR_TEAM_GAMES[year][team]}') # 2016 CHC vs PIT 무승부 처리됨
    finally: 
        if DEBUG:
            print(team, year, len(games), 'okay')
    result_df = games_df[['game_id', 'game_date', 'away_id', 'home_id', 'doubleheader', 'game_num', 'summary', 'away_score', 'home_score', 'away_name', 'home_name']] # 풀로 하면 경기 설명에서 개행때문에 제대로 write가 안되는 문제 발생
    result_df['home_win'] = result_df.home_score > result_df.away_score # warning
    return result_df

def add_player_name(df):
    # add pitcher, batter name from playerid_lookup
    # assert df.game_type == 'R'
    pitcher_df = playerid_reverse_lookup(df.pitcher.unique())
    batter_df = playerid_reverse_lookup(df.batter.unique())
    
    # breakpoint()
    player_df = pd.concat([pitcher_df, batter_df], axis=0)

    player_df['name'] = player_df['name_first'].str.capitalize() + ' ' + player_df['name_last'].str.capitalize()
    player_df['name_comma'] = player_df['name_last'].str.capitalize() + ', ' + player_df['name_first'].str.capitalize()

    for pid, name, cname in tqdm(zip(player_df.key_mlbam, player_df.name, player_df.name_comma)): ############### length check 필요 (progress bar) | 4255it [01:35, 44.67it/s]
        df.loc[df.pitcher == pid, 'pitcher_name'] = name
        df.loc[df.batter == pid, 'batter_name'] = name
        df.loc[df.pitcher == pid, 'pitcher_name_comma'] = cname
        df.loc[df.batter == pid, 'batter_name_comma'] = cname
    
    pitcher_name_column = df.pop('pitcher_name')
    df.insert(df.columns.to_list().index('pitcher'), 'pitcher_name', pitcher_name_column)

    batter_name_column = df.pop('batter_name')
    df.insert(df.columns.to_list().index('batter'), 'batter_name', batter_name_column)


def add_team_name(df):
    # add team name from teamid
    # print('add team')
    gamedate_idx = df.columns.to_list().index('game_date')
    df.insert(gamedate_idx, 'home_team', df.home_id.apply(lambda x: CODE_TO_TEAM[int(x)]))
    df.insert(gamedate_idx, 'away_team', df.away_id.apply(lambda x: CODE_TO_TEAM[int(x)]))


def get_start_lineup(statcast_df, game_pk):
    # game_pk(게임 일련번호)에 해당하는 df를 읽어옴 
    # 홈팀과 원정팀의 선발타자를 반환
    # lineup_df = pybaseball.
    # statcast_df = pd.read_csv('2015-to-2021.csv')
    players = {
        'away': {
            'team': None,
            'start_pitcher_id': None, 'start_pitcher_name': None,
            'start_lineup_id': [], 'start_lineup_name': []
        }, 
        'home': {
            'team': None,
            'start_pitcher_id': None, 'start_pitcher_name': None,
            'start_lineup_id': [], 'start_lineup_name': []
        },
        'game_date' : None
    }

    # statcast_df = pd.read_csv('2015-04-05.csv') # 413661
    gameday_df = statcast_df[statcast_df.game_pk == game_pk].sort_values(by=['at_bat_number']).sort_index(ascending=False) # 공 단위 sequence로 정렬 (sort_values 필요한지 확인 필요) # 왜 수비 시프트에 NaN이 나옴? (sort_index false와 unnamed:0 로 바꿔야 하는지 체크 필요)

    players['game_date'] = gameday_df.iloc[0].game_date

    players['home']['team'] = gameday_df.iloc[0].home_team
    players['away']['team'] = gameday_df.iloc[0].away_team

    home_df = gameday_df[gameday_df.inning_topbot == 'Bot'] # 말공은 홈팀 (타자 뽑기 위함)
    away_df = gameday_df[gameday_df.inning_topbot == 'Top'] # 선공은 원정팀 (타자 뽑기 위함)

    players['away']['start_pitcher_id'] = home_df.iloc[0].pitcher
    players['away']['start_pitcher_name'] = home_df.iloc[0].pitcher_name
    players['home']['start_pitcher_id'] = away_df.iloc[0].pitcher
    players['home']['start_pitcher_name'] = away_df.iloc[0].pitcher_name

    # breakpoint()
    # auto generated. check 필요
    for iter, row in home_df.iterrows():
        # if start lineup is smaller than 9 and player is not added, add player
        if len(players['home']['start_lineup_id']) < 9 and row.batter not in players['home']['start_lineup_id']:
            players['home']['start_lineup_id'].append(row.batter)
            players['home']['start_lineup_name'].append(row.batter_name)

    # pprint(players['home'])
    assert len(set(players['home']['start_lineup_id'])) == 9, 'home start lineup is not 9'
    
    for iter, row in away_df.iterrows():
        # if start lineup is smaller than 9 and player is not added, add player
        if len(players['away']['start_lineup_id']) < 9 and row.batter not in players['away']['start_lineup_id']:
            players['away']['start_lineup_id'].append(row.batter)
            players['away']['start_lineup_name'].append(row.batter_name) # 이름 바로 나오는지 체크

    # pprint(players['away'])
    assert len(set(players['away']['start_lineup_id'])) == 9, 'away start lineup is not 9'

    return players
    # statcast_df['game_pk'] 로드
    # 해당 홈, 원정 팀의 첫 9명 타자 가져오기
    # for문으로 at_Bat_number 늘려가면서 홈 9개, 타자 9개 가져오기 
    # 그리고 그것 반환


def write_lineup_data(players_dict):
    breakpoint()

if __name__ == '__main__':
    # 팀별 연도별저장
    # start_year, end_year = 2015, 2022
    # year_team_gmkey_dict = {}
    # year_team_game_list = []
    # for year, team in product(range(start_year, end_year+1), TEAMS_TO_CODE.keys()):
    #     result = get_game_pk(team, year)
    #     year_team_gmkey_dict[(year, team)] = result # 해당 연도 해당 팀의 df
    #     year_team_game_list.append(result) # 전체 경기

    # merged_df = pd.concat(year_team_game_list) # start_year ~ end_year까지의 모든 경기

    # for year, team in product(range(start_year, end_year+1), TEAMS_TO_CODE.keys()):
    #     year_team_gmkey_dict[(year, team)].to_csv(f'team_gameinfo/{year}/{team}-game-data.csv', encoding='utf-8-sig', index=False) # 팀별로 데이터 저장
    # print(merged_df.shape)
    # merged_df.to_csv(f'{start_year}-to-{end_year}-team-data.csv', encoding='utf-8', index=False) # 전체 데이터 저장

    # try:
    #     merged_dup_df = merged_df.drop_duplicates()
    #     print(merged_dup_df.shape)
    #     merged_dup_df.to_csv(f'{start_year}-to-{end_year}-team-data-dup.csv', encoding='utf-8', index=False)
    # except Exception as e:
    #     print(e)

    # 2021 or 2022(data 다시 모아야 함)
    # filename = '2015-to-2022.csv' # '2015-04-05.csv' # play-by-play data
    # team_filename = f'2015-to-2022-team-data.csv' # 팀 승패 기록
    filename = '2015-to-2021.csv' # '2015-04-05.csv' # play-by-play data
    # 타자 및 투수 이름 추가
    team_filename = f'2015-to-2021-team-data.csv' # 팀 승패 기록
    # 팀명 추가

    pbp_data = pd.read_csv(filename)
    pbp_data = pbp_data[pbp_data.game_type == 'R'] # 정규시즌만
    add_player_name(pbp_data)

    team_data = pd.read_csv(team_filename) # 팀 승패 기록
    add_team_name(team_data)

    games_pk = np.sort(team_data.game_id.unique()) # 여기서 game_id -> 실제 데이터에서의 game_pk

    first_five, last_five = games_pk[:5], games_pk[-5:]

    games_lineup = dict()

    for game_pk in tqdm(games_pk):
        # 팀 정보 가져오게 수정?
        game_lineup_info = get_start_lineup(pbp_data, game_pk)
        if not game_pk in games_lineup:
            games_lineup[game_pk] = game_lineup_info
    
    # print game sample
    for sample_game in first_five:
        pprint(games_lineup[sample_game])
    for sample_game in last_five:
        pprint(games_lineup[sample_game])

    # 졌는지 이겼는지 체크
    write_lineup_data(games_lineup)

'''
현재 에러나는 팀.연도                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

2015 TOR 
2016 TB TOR
2017 TOR

-> 모듈 수정으로 해결

{'Completed Early', 'Completed Early: Rain', 'Final: Tied', 'Postponed', 'Final', 'Cancelled'}
1) 코로나 시즌 더블헤더 7이닝의 두번째 경기
2) 우천으로 인한 경기 서스펜디드
3) 무승부 처리 (2016년 1번)
'''


# 그냥 연도별로 조회해서 합하고 df로 한번에 저장하기 -> 중복 없애고