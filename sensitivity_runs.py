import killing_agents as ka
import resurrecting_agents as ra
import networkx as nx
import pickle
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm

file = open('data_sensitivity/close_stops.pkl', 'rb')
close_stops = pickle.load(file)
file.close()

am_counts_df_200_500 = pd.read_csv('data_sensitivity/am_counts_200_500.csv', index_col=0)


speeds_0 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 'TRAM': 1, 'BUS': 1, 'FILOBUS': 1, 'foot': 1}
speeds_1 = {1: 32, 2: 32, 3: 32, 4: 32, 5: 32, 'TRAM': 11, 'BUS': 14, 'FILOBUS': 14, 'foot': 5}
speeds_2 = {1: 32, 2: 32, 3: 32, 4: 32, 5: 32, 'TRAM': 14, 'BUS': 14, 'FILOBUS': 14, 'foot': 5} 
speeds = pd.concat([pd.Series(speeds_0), pd.Series(speeds_1), pd.Series(speeds_2)], 1).T


G1 = nx.read_gpickle('data_sensitivity/G1')
G2 = nx.read_gpickle('data_sensitivity/G2')
G3 = nx.read_gpickle('data_sensitivity/G3')
G4 = nx.read_gpickle('data_sensitivity/G4')
G5 = nx.read_gpickle('data_sensitivity/G5')
G6 = nx.read_gpickle('data_sensitivity/G6')
G7 = nx.read_gpickle('data_sensitivity/G7')
G8 = nx.read_gpickle('data_sensitivity/G8')

outfile = open("data_sensitivity/foot1",'rb')
foot1 = pickle.load(outfile)
outfile.close()
outfile = open("data_sensitivity/foot2",'rb')
foot2 = pickle.load(outfile)
outfile.close()
outfile = open("data_sensitivity/foot3",'rb')
foot3 = pickle.load(outfile)
outfile.close()
outfile = open("data_sensitivity/foot4",'rb')
foot4 = pickle.load(outfile)
outfile.close()
outfile = open("data_sensitivity/foot5",'rb')
foot5 = pickle.load(outfile)
outfile.close()
outfile = open("data_sensitivity/foot6",'rb')
foot6 = pickle.load(outfile)
outfile.close()
outfile = open("data_sensitivity/foot7",'rb')
foot7 = pickle.load(outfile)
outfile.close()
outfile = open("data_sensitivity/foot8",'rb')
foot8 = pickle.load(outfile)
outfile.close()

graphs = [G1, G2, G3, G4, G5, G6, G7, G8]
foots = [foot1, foot2, foot3, foot4, foot5, foot6, foot7, foot8]

settings_complete = pd.DataFrame.from_dict({ 1: {'foot_r': 100, 'speeds' : 1, 'am_counts_df' : 200, 'p_long' : 0.7, 'agents_strategy' : 1},
                                             2: {'foot_r': 100, 'speeds' : 2, 'am_counts_df' : 200, 'p_long' : 0.7, 'agents_strategy' : 1},
                                             3: {'foot_r': 100, 'speeds' : 1, 'am_counts_df' : 500, 'p_long' : 0.7, 'agents_strategy' : 1},
                                             4: {'foot_r': 100, 'speeds' : 2, 'am_counts_df' : 500, 'p_long' : 0.7, 'agents_strategy' : 1},
                                             5: {'foot_r': 200, 'speeds' : 1, 'am_counts_df' : 200, 'p_long' : 0.7, 'agents_strategy' : 1},
                                             6: {'foot_r': 200, 'speeds' : 2, 'am_counts_df' : 200, 'p_long' : 0.7, 'agents_strategy' : 1},
                                             7: {'foot_r': 200, 'speeds' : 1, 'am_counts_df' : 500, 'p_long' : 0.7, 'agents_strategy' : 1},
                                             8: {'foot_r': 200, 'speeds' : 2, 'am_counts_df' : 500, 'p_long' : 0.7, 'agents_strategy' : 1},
                                             9: {'foot_r': 100, 'speeds' : 1, 'am_counts_df' : 200, 'p_long' : 0.6, 'agents_strategy' : 1},
                                            10: {'foot_r': 100, 'speeds' : 2, 'am_counts_df' : 200, 'p_long' : 0.6, 'agents_strategy' : 1},
                                            11: {'foot_r': 100, 'speeds' : 1, 'am_counts_df' : 500, 'p_long' : 0.6, 'agents_strategy' : 1},
                                            12: {'foot_r': 100, 'speeds' : 2, 'am_counts_df' : 500, 'p_long' : 0.6, 'agents_strategy' : 1},
                                            13: {'foot_r': 200, 'speeds' : 1, 'am_counts_df' : 200, 'p_long' : 0.6, 'agents_strategy' : 1},
                                            14: {'foot_r': 200, 'speeds' : 2, 'am_counts_df' : 200, 'p_long' : 0.6, 'agents_strategy' : 1},
                                            15: {'foot_r': 200, 'speeds' : 1, 'am_counts_df' : 500, 'p_long' : 0.6, 'agents_strategy' : 1},
                                            16: {'foot_r': 200, 'speeds' : 2, 'am_counts_df' : 500, 'p_long' : 0.6, 'agents_strategy' : 1},
                                            17: {'foot_r': 100, 'speeds' : 1, 'am_counts_df' : 200, 'p_long' : 0.5, 'agents_strategy' : 1},
                                            18: {'foot_r': 100, 'speeds' : 2, 'am_counts_df' : 200, 'p_long' : 0.5, 'agents_strategy' : 1},
                                            19: {'foot_r': 100, 'speeds' : 1, 'am_counts_df' : 500, 'p_long' : 0.5, 'agents_strategy' : 1},
                                            20: {'foot_r': 100, 'speeds' : 2, 'am_counts_df' : 500, 'p_long' : 0.5, 'agents_strategy' : 1},
                                            21: {'foot_r': 200, 'speeds' : 1, 'am_counts_df' : 200, 'p_long' : 0.5, 'agents_strategy' : 1},
                                            22: {'foot_r': 200, 'speeds' : 2, 'am_counts_df' : 200, 'p_long' : 0.5, 'agents_strategy' : 1},
                                            23: {'foot_r': 200, 'speeds' : 1, 'am_counts_df' : 500, 'p_long' : 0.5, 'agents_strategy' : 1},
                                            24: {'foot_r': 200, 'speeds' : 2, 'am_counts_df' : 500, 'p_long' : 0.5, 'agents_strategy' : 1},
                                            25: {'foot_r': 100, 'speeds' : 1, 'am_counts_df' : 200, 'p_long' : 0.7, 'agents_strategy' : 2},
                                            26: {'foot_r': 100, 'speeds' : 2, 'am_counts_df' : 200, 'p_long' : 0.7, 'agents_strategy' : 2},
                                            27: {'foot_r': 100, 'speeds' : 1, 'am_counts_df' : 500, 'p_long' : 0.7, 'agents_strategy' : 2},
                                            28: {'foot_r': 100, 'speeds' : 2, 'am_counts_df' : 500, 'p_long' : 0.7, 'agents_strategy' : 2},
                                            29: {'foot_r': 200, 'speeds' : 1, 'am_counts_df' : 200, 'p_long' : 0.7, 'agents_strategy' : 2},
                                            30: {'foot_r': 200, 'speeds' : 2, 'am_counts_df' : 200, 'p_long' : 0.7, 'agents_strategy' : 2},
                                            31: {'foot_r': 200, 'speeds' : 1, 'am_counts_df' : 500, 'p_long' : 0.7, 'agents_strategy' : 2},
                                            32: {'foot_r': 200, 'speeds' : 2, 'am_counts_df' : 500, 'p_long' : 0.7, 'agents_strategy' : 2},
                                            33: {'foot_r': 100, 'speeds' : 1, 'am_counts_df' : 200, 'p_long' : 0.6, 'agents_strategy' : 2},
                                            34: {'foot_r': 100, 'speeds' : 2, 'am_counts_df' : 200, 'p_long' : 0.6, 'agents_strategy' : 2},
                                            35: {'foot_r': 100, 'speeds' : 1, 'am_counts_df' : 500, 'p_long' : 0.6, 'agents_strategy' : 2},
                                            36: {'foot_r': 100, 'speeds' : 2, 'am_counts_df' : 500, 'p_long' : 0.6, 'agents_strategy' : 2},
                                            37: {'foot_r': 200, 'speeds' : 1, 'am_counts_df' : 200, 'p_long' : 0.6, 'agents_strategy' : 2},
                                            38: {'foot_r': 200, 'speeds' : 2, 'am_counts_df' : 200, 'p_long' : 0.6, 'agents_strategy' : 2},
                                            39: {'foot_r': 200, 'speeds' : 1, 'am_counts_df' : 500, 'p_long' : 0.6, 'agents_strategy' : 2},
                                            40: {'foot_r': 200, 'speeds' : 2, 'am_counts_df' : 500, 'p_long' : 0.6, 'agents_strategy' : 2},
                                            41: {'foot_r': 100, 'speeds' : 1, 'am_counts_df' : 200, 'p_long' : 0.5, 'agents_strategy' : 2},
                                            42: {'foot_r': 100, 'speeds' : 2, 'am_counts_df' : 200, 'p_long' : 0.5, 'agents_strategy' : 2},
                                            43: {'foot_r': 100, 'speeds' : 1, 'am_counts_df' : 500, 'p_long' : 0.5, 'agents_strategy' : 2},
                                            44: {'foot_r': 100, 'speeds' : 2, 'am_counts_df' : 500, 'p_long' : 0.5, 'agents_strategy' : 2},
                                            45: {'foot_r': 200, 'speeds' : 1, 'am_counts_df' : 200, 'p_long' : 0.5, 'agents_strategy' : 2},
                                            46: {'foot_r': 200, 'speeds' : 2, 'am_counts_df' : 200, 'p_long' : 0.5, 'agents_strategy' : 2},
                                            47: {'foot_r': 200, 'speeds' : 1, 'am_counts_df' : 500, 'p_long' : 0.5, 'agents_strategy' : 2},
                                            48: {'foot_r': 200, 'speeds' : 2, 'am_counts_df' : 500, 'p_long' : 0.5, 'agents_strategy' : 2}}, orient = 'index')


for i in tqdm(settings_complete.index[:1]):
    ind = i%8-1
    G = copy.deepcopy(graphs[ind])
    am = int(settings_complete.loc[i]['am_counts_df'])
    am_counts_df_ = am_counts_df_200_500[[f'leisure_{str(am)}',f'food_{str(am)}',f'school_{str(am)}',f'office_{str(am)}']].rename(columns = {f'leisure_{str(am)}':'leisure',f'food_{str(am)}':'food',f'school_{str(am)}':'school',f'office_{str(am)}':'office'})
    
    am_counts_df_prob = am_counts_df_ / am_counts_df_.sum(axis = 0) 

    if settings_complete.loc[i]['agents_strategy'] == 1:
        n = 25000
        # n = 5
    elif settings_complete.loc[i]['agents_strategy'] == 2:
        n = 20000
        # n = 5

    s = ka.simulation2(G, n, foot=foots[ind], am_counts_df_prob = am_counts_df_prob, agents_strategy = settings_complete.loc[i]['agents_strategy'], p_long = settings_complete.loc[i]['p_long'])
    ra.save_experiment(s, f'prova_s/setting{i}_')

    if i == 1:
        r = ka.simulation_results(s[0], s[1], G, run= i, results=False)
    else:
        r = ka.simulation_results(s[0], s[1], G, run = i, results = r)


    print(f'Run {i} completed!')

r.to_csv(f'prova_s/final_results.csv')