# %%
from collections import Counter
from pprint import pprint 
import sys 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from pybaseball import playerid_reverse_lookup
from tqdm import tqdm

model_type = 'normal' # ['normal', 'multilabel', 'simplified']
start_year, end_year = 2015, 2021
is_dropout = False
epochs = 500
lr = 0.0001
batch_size = 4096 # len(x_input)
n_dim = 16
weight_decay = 1e-4
delta_uniformity = 0.1
limit = [5000, 1000, 500][int(sys.argv[1])] if len(sys.argv) >= 2 else 3000 # 1000구 이상 대결한 선수들

print(f'load {start_year}-{end_year} data...')
data = pd.read_csv('2015-to-2021.csv', encoding='utf8') # read file
data = data[data['game_type'] == 'R'] # only regular season
data = data[data['game_date'].str.startswith(tuple(map(str,range(start_year, end_year+1))))]
print(f'load {start_year}-{end_year} data...')

# effective_col = ('pitch_type release_speed release_pos_x release_pos_z batter pitcher events zone stand p_throws type hit_location bb_type ' +
# 'pfx_x pfx_z plate_x plate_z hit_distance_sc launch_speed launch_angle release_spin_rate').split()
# %%
triplet = data['batter pitcher events hit_location bb_type'.split()]
triplet.hit_location = triplet.hit_location.fillna(0).astype(int) # NA as 0

# %%
print(f'triplet hit location : {Counter(triplet.hit_location)}') # 타구방향
print(f'triplet events : {Counter(triplet.events)}') # 결과

# %%
new_triplet = triplet.assign(events_hit_location=triplet.events.map(str) + '_' + \
                triplet.hit_location.map(str)) # make label

# %%
# make label ((batter|pitcher)2vec)
event_cnt = Counter(new_triplet.events_hit_location)
events = sorted(event_cnt.items(), key=lambda x:x[1], reverse=True)

events_label_dict = {
    'normal' : dict(),
    'multilabel' : dict(), 
    'simplified' : dict()
}

for event, n_event in events:
    if event.startswith('strikeout'):
        events_label_dict['normal'][event] = 'K'
        events_label_dict['multilabel'][event] = ('K', '-1')
        events_label_dict['simplified'][event] = 'K'
        # events_to_label[event] = 'K'
        # events_to_multilabel[event] = ('K', '-1')
        # events_to_simplified_label[event] = 'K'
    elif event.startswith('walk'):
        events_label_dict['normal'][event] = 'BB'
        events_label_dict['multilabel'][event] = ('BB', '-1')
        events_label_dict['simplified'][event] = 'BBHP'
        # events_to_label[event] = 'W'
        # events_to_multilabel[event] = ('W', '-1')
        # events_to_simplified_label[event] = 'BBHP'
    elif event.startswith('intent_walk'):
        events_label_dict['normal'][event] = 'IBB'
        events_label_dict['multilabel'][event] = ('IBB', '-1')
        events_label_dict['simplified'][event] = 'BBHP'
        # events_to_label[event] = 'IW'
        # events_to_multilabel[event] = ('IW', '-1')
        # events_to_simplified_label[event] = 'BBHP'
    elif event.startswith('balk'): # not reached
        events_label_dict['normal'][event] = 'BK'
        events_label_dict['multilabel'][event] = ('BK', '-1')
        events_label_dict['simplified'][event] = 'BK'
        # events_to_label[event] = 'BK'
        # events_to_multilabel[event] = ('BK', '-1')
        # events_to_simplified_label[event] = 'BK'
    elif event.startswith('hit_by_pitch'):
        events_label_dict['normal'][event] = 'HBP'
        events_label_dict['multilabel'][event] = ('HBP', '-1')
        events_label_dict['simplified'][event] = 'BBHP'
        # events_to_label[event] = 'HP'
        # events_to_multilabel[event] = ('HP', '-1')
        # events_to_simplified_label[event] = 'BBHP'

    elif event.startswith((
        'field_out', 'force_out', 'fielders_choice_out', 'other_out', 
        'double_play', 'grounded_into_double_play', 'triple_play',
        'sac_fly', 'sac_bunt', 'fielders_choice')):
        events_label_dict['normal'][event] = event[-1]
        events_label_dict['multilabel'][event] = ('out', event[-1])
        events_label_dict['simplified'][event] = 'out'
        # events_to_label[event] = event[-1]
        # events_to_multilabel[event] = ('out', event[-1])
        # events_to_simplified_label[event] = 'out'
    elif event.startswith('field_error'):
        events_label_dict['normal'][event] = 'E'+event[-1]
        events_label_dict['multilabel'][event] = ('E', event[-1])
        events_label_dict['simplified'][event] = 'out' # 혹은 무시
        # events_to_label[event] = 'E'+event[-1]
        # events_to_multilabel[event] = ('E', event[-1])
        # # events_to_simplified_label[event] = 'out' # 혹은 무시
    elif event.startswith('single'):
        events_label_dict['normal'][event] = 'S'+event[-1]
        events_label_dict['multilabel'][event] = ('S', event[-1])
        events_label_dict['simplified'][event] = '1B'
        # events_to_label[event] = 'S'+event[-1]
        # events_to_multilabel[event] = ('S', event[-1])
        # events_to_simplified_label[event] = 'H'
    elif event.startswith('double_0'): # csv 불러올때 17개는 fan interference
        events_label_dict['normal'][event] = 'DG'
        events_label_dict['multilabel'][event] = ('DG', '10')
        events_label_dict['simplified'][event] = '2B'
        # events_to_label[event] = 'DG' # ground rule double 이지만 des에 해당 글이 있는지 체크 필요
        # events_to_multilabel[event] = ('DG', '10') # only for DG (0과 다르게 처리하기 위함)
        # events_to_simplified_label[event] = '2B'
    elif event.startswith('double'):
        events_label_dict['normal'][event] = 'D'+event[-1]
        events_label_dict['multilabel'][event] = ('D', event[-1])
        events_label_dict['simplified'][event] = '2B'
        # events_to_label[event] = 'D'+event[-1]
        # events_to_multilabel[event] = ('D', event[-1])
        # events_to_simplified_label[event] = '2B'
    elif event.startswith('triple'):
        events_label_dict['normal'][event] = 'T'+event[-1]
        events_label_dict['multilabel'][event] = ('T', event[-1])
        events_label_dict['simplified'][event] = '3B'
        # events_to_label[event] = 'T'+event[-1]
        # events_to_multilabel[event] = ('T', event[-1])
        # events_to_simplified_label[event] = '3B'
    elif event.startswith('home_run'):
        events_label_dict['normal'][event] = 'HR'
        events_label_dict['multilabel'][event] = ('HR', event[-1])
        events_label_dict['simplified'][event] = 'HR'
        # events_to_label[event] = 'HR'
        # events_to_multilabel[event] = ('HR', event[-1]) # check 필요
        # events_to_simplified_label[event] = 'HR'

    elif event.startswith('nan'):
        pass # no event occured
    elif event.startswith(('caught_stealing', 'pickoff', 'runner_double_play', 'stolen_base')):
        pass # not a batter event
    elif event.startswith(('passed_ball', 'wild_pitch', 'catcher_interf')):
        pass # not a batter event
    elif event.startswith(('ejection', 'game_advisory')):
        pass # not a batter event
    else:
        print('not classified', event, 'n_event =', event_cnt[event])
    print(event, n_event, events_label_dict['simplified'][event] if event in events_label_dict['simplified'] else 'not classified')

# %%

events_to_label = events_label_dict['normal']
events_to_multilabel = events_label_dict['multilabel']
events_to_simplified_label = events_label_dict['simplified']

print(f'{events_to_label=}')
print(f'{events_to_multilabel=}')
print(f'{events_to_simplified_label=}')

labels = sorted(list(set(events_to_label.values()))) # 49 labels (2018~2021)
label_to_idx = {label:idx for idx, label in enumerate(labels)}
n_result = len(labels)
print(f'{label_to_idx=}')

labels_outcome = sorted(list(set([label for label, _ in events_to_multilabel.values()])))
label_outcome_to_idx = {label:idx for idx, label in enumerate(labels_outcome)}
labels_direction = sorted(list(set([label for _, label in events_to_multilabel.values()])), key=lambda x:int(x))
label_direction_to_idx = {label:idx for idx, label in enumerate(labels_direction)}
n_result_multilabel = (len(labels_outcome), len(labels_direction))

print(f'{labels_outcome=}')
print(f'{labels_direction=}')
print(f'{label_outcome_to_idx=}')
print(f'{label_direction_to_idx=}')

labels_smp = sorted(list(set(events_to_simplified_label.values())))
label_smp_to_idx = {label:idx for idx, label in enumerate(labels_smp)}
n_result_smp = len(labels_smp)
print(f'{label_smp_to_idx=}')

# %%
# some weird events
# print(data[(data.events == 'single') & (data.hit_location.isnull())]['game_date player_name batter pitcher events hit_location bb_type'.split()])

# %%
def get_player_info(player_data, ball_limit=1000):
    player_vc = player_data.value_counts()
    probable_players = player_vc[player_vc > ball_limit].index
    df = playerid_reverse_lookup(probable_players)
    name_last, name_first = df['name_last'], df['name_first']
    df['full_name'] = name_first.str.capitalize() + ' ' + name_last.str.capitalize()
    return df[['key_mlbam', 'full_name']]

def get_player_name(player_db, key): # 타자 및 투수 합해야?
    name_df = player_db[player_db.key_mlbam == key].full_name
    if name_df.empty:
        # raise Exception(f'no player name found for {key}')
        return None
    else:
        return name_df.values[0]

prob_batters = get_player_info(data['batter'], limit)
prob_pitchers = get_player_info(data['pitcher'], limit)

batter_key_to_idx = dict(zip(prob_batters.key_mlbam, range(len(prob_batters))))
pitcher_key_to_idx = dict(zip(prob_pitchers.key_mlbam, range(len(prob_pitchers))))

batter_idx_to_key = {idx:key for key, idx in batter_key_to_idx.items()}
pitcher_idx_to_key = {idx:key for key, idx in pitcher_key_to_idx.items()}

batter_key_to_name = dict(zip(prob_batters.key_mlbam, prob_batters.full_name))
pitcher_key_to_name = dict(zip(prob_pitchers.key_mlbam,
 prob_pitchers.full_name))

batter_name_to_key = {name:key for key, name in batter_key_to_name.items()}
pitcher_name_to_key = {name:key for key, name in pitcher_key_to_name.items()}

print(prob_batters)
print(prob_pitchers)
# breakpoint()

# print(get_player_name(prob_batters, 660271))
# print(get_player_name(prob_pitchers, 660271))


# %%
batter_np = prob_batters.key_mlbam.to_numpy() # 타자들 id
pitcher_np = prob_pitchers.key_mlbam.to_numpy() # 투수들 id

x_inputs = {'normal':[], 'multilabel': [], 'simplified':[]} # 기본, 타구/방향, 단순화 label

print('preprocessing data: ')
# events_to_label, events_to_multilabel, events_to_simplified_label 로 나누어 데이터 저장
for line in tqdm(new_triplet[['batter', 'pitcher', 'events_hit_location']].to_numpy()):
    # print(line)
    if line[0] in batter_np and line[1] in pitcher_np:
        if line[-1] in events_to_label:
            batter_idx = batter_key_to_idx[line[0]]
            pitcher_idx = pitcher_key_to_idx[line[1]]

            label_idx = label_to_idx[events_to_label[line[-1]]]
            x_inputs['normal'].append([batter_idx, pitcher_idx, label_idx])

            label_idx_smp = label_smp_to_idx[events_to_simplified_label[line[-1]]]
            x_inputs['simplified'].append([batter_idx, pitcher_idx, label_idx_smp])

            label_idx_outcome, label_idx_direction = label_outcome_to_idx[events_to_multilabel[line[-1]][0]], label_direction_to_idx[events_to_multilabel[line[-1]][1]]
            x_inputs['multilabel'].append([batter_idx, pitcher_idx, label_idx_outcome, label_idx_direction])


# %%
class Batter_Pitcher_2_Vec(nn.Module):
    def __init__(self, n_batter, n_pitcher, n_dim, n_result, device='gpu', dropout=False, multi_label=False):
        super().__init__()
        self.device = 'cuda' if device == 'gpu' else 'cpu'
        self.multi_label = multi_label
        # self.player_dict = player_dict # 'batter': tensor of player id with shape (n_batter, ), 'pitcher': tensor of player id with shape (n_pitcher, )
        self.batter_emb = nn.Embedding(n_batter, n_dim)
        self.pitcher_emb = nn.Embedding(n_pitcher, n_dim)

        if self.multi_label:
            self.linear_outcome = nn.Linear(n_dim, n_result[0])
            self.linear_direction = nn.Linear(n_dim, n_result[1])
        else:
            self.linear = nn.Linear(2*n_dim, n_result)
        self.dropout = nn.Dropout(0.5) if dropout else None
        
    def forward(self, x): 
        batter_idx, pitcher_idx = x[:,0], x[:,1]

        batter_emb = self.dropout(self.batter_emb(batter_idx)) if self.dropout else self.batter_emb(batter_idx)
        pitcher_emb = self.dropout(self.pitcher_emb(pitcher_idx)) if self.dropout else self.pitcher_emb(pitcher_idx)

        merge_emb = torch.cat([batter_emb, pitcher_emb], dim=1)

        if self.multi_label:
            outcome = self.linear_outcome(merge_emb)
            direction = self.linear_direction(merge_emb)
            return (outcome, direction), batter_emb, pitcher_emb
        else:
            y_pred = self.linear(merge_emb)
            return y_pred, batter_emb, pitcher_emb

# %%
def alignment(x, y):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(2).mean()

def uniformity(x):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

def calculate_loss(self, user, item):
    user_e, item_e = self.encoder(user, item)  # [bsz, dim]
    align = self.alignment(user_e, item_e)
    uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
    loss = align + self.gamma * uniform
    return loss

# %%
# import TSNE
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

def visualize(embedding, player_list, fname):
    plt.clf()
    embedding = F.normalize(embedding, dim=-1)
    embedding = embedding.detach().cpu().numpy()
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(embedding)
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()
    plt.savefig(fname)

def nearest_neighbor(embedding, player_name, player_type, n=10):
    embedding = F.normalize(embedding, dim=-1)
    embedding = embedding.detach().cpu().numpy()

    if player_type == 'b':
        player_id = batter_name_to_key[player_name]
        idx = batter_key_to_idx[player_id]
    elif player_type == 'p':
        player_id = pitcher_name_to_key[player_name]
        idx = pitcher_key_to_idx[player_id]

    player_emb = embedding[idx]
    dist = np.linalg.norm(embedding - player_emb, axis=1) # cossim 으로 바꾸기
    dist[idx] = np.inf
    sorted_dist = sorted(zip(range(len(dist)), dist), key=lambda x: x[1])[:n]
    # idx to player_id to player name
    result_list = []
    for nearest_idx, norm in sorted_dist:
        if player_type == 'b':
            result_list.append((batter_key_to_name[batter_idx_to_key[nearest_idx]], norm))
        elif player_type == 'p':
            result_list.append((pitcher_key_to_name[pitcher_idx_to_key[nearest_idx]], norm))
    print(f'Nearest Neighbor of {player_name}:\t', result_list)
    return result_list

# %%
x_input = torch.from_numpy(x_inputs[model_type]).long() # 데이터 결정

if model_type == 'normal':
    model = Batter_Pitcher_2_Vec(len(batter_key_to_idx), len(pitcher_key_to_idx), n_dim, n_result, dropout=is_dropout)
elif model_type == 'multilabel':
    model = Batter_Pitcher_2_Vec(len(batter_key_to_idx), len(pitcher_key_to_idx), n_dim, n_result_multilabel, multi_label=True, dropout=is_dropout)
elif model_type == 'simplified':
    model = Batter_Pitcher_2_Vec(len(batter_key_to_idx), len(pitcher_key_to_idx), n_dim, n_result_smp, dropout=is_dropout)

model.cuda()
data_loader = DataLoader(x_input, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
model.train()

for epoch in range(epochs):
    loss = 0        
    for x_batch in tqdm(data_loader):
        x_batch = x_batch.cuda()
        model_output, batter_emb, pitcher_emb = model(x_batch)
        batch_loss = criterion(model_output, x_batch[:,2]) # + delta_uniformity * (uniformity(batter_emb) + uniformity(pitcher_emb))
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item() / len(x_batch) # avg
    print(f'epoch: {epoch}, loss: {loss}')
    if loss == np.nan:
        breakpoint()

    # 더 많은 예시로 뽑아보기 (샘플 많은 선수, 중간 선수, 적은 선수 변화량 분석)
    # breakpoint()
    print('Embedding of batter Shohei Ohtani: ', np.around(model.batter_emb.weight[batter_key_to_idx[660271]].detach().cpu().numpy(), 2))
    print('Embedding of batter Jose Ramirez: ', np.around(model.batter_emb.weight[batter_key_to_idx[608070]].detach().cpu().numpy(), 2))
    print('Embedding of batter Jose Altuve: ', np.around(model.batter_emb.weight[batter_key_to_idx[514888]].detach().cpu().numpy(), 2))
    print('Embedding of pitcher Max Scherzer: ', np.around((model.pitcher_emb.weight[pitcher_key_to_idx[453286]].detach().cpu().numpy()), 2))
    print('Embedding of pitcher Ryan Yarbrough: ', np.around((model.pitcher_emb.weight[pitcher_key_to_idx[642232]].detach().cpu().numpy()), 2))
    print('Embedding of pitcher Kenley Jansen: ', np.around((model.pitcher_emb.weight[pitcher_key_to_idx[445276]].detach().cpu().numpy()), 2))
    nearest_neighbor(model.batter_emb.weight, 'Shohei Ohtani', 'b')
    nearest_neighbor(model.batter_emb.weight, 'Jose Ramirez', 'b')
    nearest_neighbor(model.batter_emb.weight, 'Jose Altuve', 'b')
    nearest_neighbor(model.pitcher_emb.weight, 'Max Scherzer', 'p')
    nearest_neighbor(model.pitcher_emb.weight, 'Ryan Yarbrough', 'p')
    nearest_neighbor(model.pitcher_emb.weight, 'Kenley Jansen', 'p')
    # element별 delta 분석?

    # 데이터가 많아서 그런지?
        # TODO 포스트시즌 데이터 제거하기
    # 상당히 빠른 시점에서 수렴하는 것 같음 -> 모델이 간단해서인지, 아니면 데이터 문제인지 확인 필요
    # (9차원 기준, 32차원도 마찬가지같음) 임베딩이 상당히 극으로 가는데 이걸 어떻게 수정해?
        # 모델에서 sigmoid 빼니까 극으로 가지는 않음.
        # 일단 dropout을 해보자
        # 차원도 늘려보고
        # gradient가 똑같아서 각각의 element가 똑같은 속도로 이동하는지 체크
            # CE 이외의 embedding에 영향을 주는 loss 추가해보기
                # uniform loss -> weight 없이 적용시 nan 뜸
                # alignment loss -> 당췌 어떻게 적용해야 함?
    # dropout? add uniformity loss? CELoss with sampling?
    # 시각화 및 neighbor search 추가해보기
    # TODO : nearest neighbor
    # print('neighbor of batter Shohei Ohtani: ', ______)

with torch.no_grad():
    model.eval()
    pass

visualize(model.batter_emb.weight, prob_batters, 'batter.png')
visualize(model.pitcher_emb.weight, prob_pitchers, 'pitcher.png')

'''
import tqdm

train_loader = tqdm.tqdm(train_dataset, desc='Loading train dataset')
for i, x in enumerate(train_loader):
	...
	...

	train_loader.set_description("Loss %.04f Acc %.04f | step %d" % (loss, acc, i))

TODO visualization
'''