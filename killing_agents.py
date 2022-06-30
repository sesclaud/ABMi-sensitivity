from logging import raiseExceptions
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import matplotlib.cm as cm
import seaborn as sns
import networkx as nx
import random
from itertools import combinations
# from geopy import distance
from tqdm.notebook import tqdm
import time
from numpy.random import multinomial
from operator import itemgetter
import itertools
from datetime import datetime, timedelta
from collections import defaultdict
import pickle

#pickles

file = open('data/coords.pkl', 'rb')
coords = pickle.load(file)
file.close()

file = open('data/percorsi', 'rb')
percorsi = pickle.load(file)
file.close()

file = open('data/foot', 'rb')
foot = pickle.load(file)
file.close()

file = open('data/d_active_edges', 'rb')
d_active_edges = pickle.load(file)
file.close()

d_ages = {0: '15-24 anni', 1: '25-44 anni', 2: '45-64 anni', 3: '65 anni e più'}

d_activities = {'spostamenti da/per il lavoro': 'lavoro retribuito',
                'spostamenti per istruzione e studio':'istruzione e formazione',
                'spostamenti per la cura della persona': 'dormire, mangiare e altra cura della persona',
                'spostamenti per lavoro familiare':'lavoro familiare',
                'spostamenti per tempo libero': 'tempo libero' }
#csv
age_distro = pd.read_csv('data/age_distro.csv')
nil_age = pd.read_csv('data/nil_age.csv')
fermate_nils = pd.read_csv('data/fermate_nils_mani.csv')
activities = pd.read_csv('data/activities.csv')
am_counts_df_prob = pd.read_csv('data/am_counts_df_prob.csv')
am_counts_df = pd.read_csv('data/am_counts_df_mani.csv')
rescaled_ts = pd.read_csv('data/rescaled_ts.csv')
orari = pd.read_csv('data/orari.csv')
fermate = pd.read_csv('data/fermate.csv')


############################################################################################################
################################# TIME CONVERSION FUNCTIONS ################################################
############################################################################################################

def str_to_minutes(x):
    return datetime.strptime(x, '%H:%M:%S').time().hour*60 + datetime.strptime(x, '%H:%M:%S').time().minute

def minutes_to_str(x):
    h = str(x//60)
    m = str(x%60)
    if len(h) == 1:
        h = '0' + h
    if len(m) == 1:
        m = '0' + m
    return f'{h}:{m}'


####################################################################################################
################################### AGENTS' GENERATION FUNCTIONS ###################################
####################################################################################################


# generate dictionary and dataframe of agents
def gen_agent(n, d_ages = d_ages, d_activities = d_activities, age_distro = age_distro, nil_age = nil_age, fermate_nils = fermate_nils, 
                activities = activities, am_counts_df_prob = am_counts_df_prob, am_counts_df = am_counts_df, rescaled_ts = rescaled_ts):
    """
    Generating an agent in the following way:
    Draw an age class, conditionally on which a NIL is drawn plus a random stop inside it as the agent's home.
    Secondly, the activities the agent will carry out are drawn independently from one another, which again are mapped to stops
    on the network proportionally to the number of amenities corresponding to that activity present in each stop.
    Finally, the departure times for these activities are drawn.
    """
    all_ages = []
    all_nils = []
    all_homes = []
    all_activities = []
    
    age = multinomial(n, age_distro.Proportion) # n independent multinomial draws
    age_nonzero = age.nonzero()[0] # position in age^ maps to age group
    age_count = age[age>0] # how many agents per age group
    ids_ages =  np.hstack([[x] * y for x, y in zip(age_nonzero, age_count)]) # joins the information of the previous 2 arrays
    all_ages = [d_ages[x] for x in ids_ages] # retrieves n age groups
    
    for age_id, count_by_age in zip(age_nonzero, age_count): # iterating to draw NILs and schedules age by age
        age_class = d_ages[age_id] # current age group
        
        nil_by_age_df = nil_age[nil_age["Età (classi funzionali)"] == age_class].reset_index(drop = True) # NILs for curr. age group
        nil = multinomial(count_by_age, nil_by_age_df.nil_distro) # as many multinomial draws of NILs as agents from curr. age
        nil_nonzero = nil.nonzero()[0] # position in nil^ maps to NIL
        nil_count = nil[nil>0] # how many agents living in each NIL in this age group
        ids_nils =  np.hstack([[x] * y for x, y in zip(nil_nonzero, nil_count)]) # joins the information of the previous 2 arrays
        nils = nil_by_age_df.loc[ids_nils, "NIL (Nuclei di Identità Locale)"].reset_index(drop = True).sample(frac = 1).tolist() # retrieves list of NILs
        all_nils += nils # add them to the list of all NILs
        
        for index, i in enumerate(nil_nonzero):
            nil_ = nil_by_age_df.loc[i, "NIL (Nuclei di Identità Locale)"]
            if nil_ != "CHIARAVALLE":
                homes = np.random.choice(fermate_nils[fermate_nils["nil"] == nil_].stop_id,nil_count[index])
            else:
                homes =  np.random.choice(fermate_nils[fermate_nils["nil"] == "PARCO DELLE ABBAZIE"].stop_id, nil_count[index])
                                          
            all_homes += list(homes)
            
        act_by_age = [[] for ind in range(count_by_age)]
        departures_by_age = [] # where departure times will be stored
        activities_df = activities[activities["Classe di età"]==age_class].sort_values("Tipo attività").reset_index(drop = True)
        
        for act in activities_df["Tipo attività"].unique():
            activity = d_activities[act]
            activities_df_ = activities_df[activities_df["Tipo attività"]== act].reset_index(drop = True)
            probability = activities_df_.Value.tolist()[0]/100
            act_ = multinomial(count_by_age, [1-probability, probability])
            num_ag = act_[1]
            act_nonzero = act_.nonzero()[0]
            act_count = act_[act_> 0]
            
            if activity != "lavoro familiare":
                act_stop = multinomial(count_by_age, am_counts_df_prob[activity] )
                act_stop_nonzero = act_stop.nonzero()[0]
                act_stop_count = act_stop[act_stop>0]
                id_act_stop = np.hstack([[j] * k for j, k in zip(act_stop_nonzero, act_stop_count)])
                act_stops = am_counts_df.loc[id_act_stop, "stops_id"].tolist() 
                random.shuffle(act_stops)
            else:
                act_stops = ["home"]*count_by_age
                        
            if num_ag > 0:
                departures_df = rescaled_ts[(rescaled_ts["Age"] == age_class) & (rescaled_ts["Activity"] == activity)].reset_index(drop = True)
                departure = multinomial(num_ag, departures_df.Density)
                departure_nonzero = departure.nonzero()[0]
                departure_count = departure[departure>0]
                id_departures = np.hstack([[j] * k for j, k in zip(departure_nonzero, departure_count)])
                departures = departures_df.loc[id_departures, "Time_lag"].tolist()
                random.shuffle(departures)
                
            id_acts = np.hstack([[x] * y for x, y in zip(act_nonzero, act_count)])
            random.shuffle(id_acts)
            counter = 0
            for i, ind in enumerate(id_acts):
                if ind == 1:
                    act_by_age[i].append([activity, str_to_minutes(departures[counter]) + random.randrange(-5,5), 
                                                     act_stops[counter]])
                    counter += 1
                    
        all_activities += act_by_age
              
    agents = np.array([all_ages, all_homes, all_nils, all_activities], dtype = object).T
    agents_df = pd.DataFrame(agents).rename(columns = {0: "Age", 1: "Home", 2: "NIL", 3: "Activities"})
    agents_df["Num_Activities"] = agents_df.Activities.apply(lambda x: len(x)) 
    agents_df = agents_df[agents_df["Num_Activities"] > 0]
    agents = agents_df.T.to_dict()
    
    return agents, agents_df



# get a path for each journey in agent's schedule
def get_a_path(start, end, t, percorsi, foot, orari, G):
    if t > 1200:
        time_lag = t + 90
    else:
        time_lag = t + 60
    
    perc_i = orari[(((orari.inizio <= t) & (orari.fine >= time_lag)) | ((orari.inizio <= t) & (orari.next_day == True))) & (orari.tipo_giorno == 'L') & (orari.linea != 151)].percorso.unique()

    G_sub = nx.MultiDiGraph()
    for i in [percorsi[k] for k in perc_i]:
        for j in i:
            G_sub.add_edge(j[0], j[1], j[2], **G.get_edge_data(j[0], j[1], j[2]))
            
    for j in foot.copy():
        G_sub.add_edge(j[0], j[1], j[2], **G.get_edge_data(j[0], j[1], j[2]))
    nodes_path = nx.shortest_path(G_sub, start, end, 'weight')
    edges_path = []
    
    for n in range(len(nodes_path[:-1])):
        if len(edges_path) == 0:
            edge = (nodes_path[::-1][n+1], nodes_path[::-1][n])
            keys_list = list(G_sub.get_edge_data(edge[0], edge[1]).keys())
            random.shuffle(keys_list)
            edges_path.append((nodes_path[::-1][n+1], nodes_path[::-1][n], keys_list[0]))
        else:
            edge = (nodes_path[::-1][n+1], nodes_path[::-1][n])
            if edges_path[-1][2] in list(G_sub.get_edge_data(edge[0], edge[1]).keys()):
                edges_path.append((nodes_path[::-1][n+1], nodes_path[::-1][n], edges_path[-1][2]))
            else:
                edge = (nodes_path[::-1][n+1], nodes_path[::-1][n])
                keys_list = list(G_sub.get_edge_data(edge[0], edge[1]).keys())
                random.shuffle(keys_list)
                edges_path.append((nodes_path[::-1][n+1], nodes_path[::-1][n], keys_list[0])) 
    return edges_path[::-1] 



# auxiliary functions to access edges attributes
def waiting_list(G, edge):
    return G[edge[0]][edge[1]][edge[2]]['waiting_list']
    
def passengers_list(G, edge):
    return G[edge[0]][edge[1]][edge[2]]['passengers_list']
    
def total_capacity(G, edge):
    return G[edge[0]][edge[1]][edge[2]]['total_capacity']

def next_edge(G, edge):
    return G[edge[0]][edge[1]][edge[2]]['next_edges']

def weight(G, edge):
    return math.ceil(G[edge[0]][edge[1]][edge[2]]['weight'])

# plot graph
def plot_graph(G, coords, figsize = (20,20), dpi = 500):
    """
    Plots graph
    """
    colors = {1: 'red', 2: 'green', 3: 'yellow', 4: 'blue', 5: 'violet', 'TRAM': 'brown', 'BUS': 'pink', 'FILOBUS': 'grey'}
    plt.figure(figsize=figsize, dpi=dpi)
    color_comp = nx.get_edge_attributes(G,'color').values()
    nx.draw_networkx(G, pos = coords, with_labels=False, node_size=1,edge_color=color_comp, arrowsize = 4, label = colors)
    plt.legend()
    

# plot specific sub graph
def plot_lines_graph(G, coords,single_lines = [], lines_group1 = [], lines_group2 = [], figsize = (20,20), dpi = 500):
    """
    Plots the newtork with only edges belonging to the specified lines. All lines in 'single_lines' are
    plotted with a different color. Lines in 'lines_group1' are all plotted in 'mediumspringgreen' and lines
    in 'lines_grou2' are all plotted in' deeppink'
    """
    colors_metro = {'METRO1': 'red', 'METRO2': 'green', 'METRO3': 'yellow', 'METRO4': 'blue', 'METRO5': 'violet'}
    colors_dict = {}
    cmap = matplotlib.cm.get_cmap('Set2')

    # colors for single lines
    if single_lines != []:
        i = 0.1
        for l in single_lines:
            if l in colors_metro:
                colors_dict[l] = colors_metro[l]
            else:
                rgba = cmap(i)
                colors_dict[l] = rgba
                i += 0.1

    # color for group one
    if lines_group1 != []:
        col1 = 'mediumspringgreen'
        for l in lines_group1:
            colors_dict[l] = col1
    
    # colors for group two
    if lines_group2 != []:
        col2 = 'deeppink'
        for l in lines_group2:
            colors_dict[l] = col2
           
    G_sub = nx.MultiDiGraph() # create sub graph
    G_sub.add_nodes_from(list(G.nodes)) # add all nodes
    
    all_lines = set(single_lines).union(set(lines_group1)).union(set(lines_group2))
    
    for edge in G.edges: # loop over all edges of original graph
        if edge[2] in set(all_lines): # only select the ones with the relevant lines
            G_sub.add_edge(edge[0], edge[1], edge[2], color = colors_dict[edge[2]]) # add them to the sub graphs
            
    plt.figure(figsize=figsize, dpi=dpi)
    colors = [G_sub[u][v][l]['color'] for u,v,l in G_sub.edges] # get colors
    nx.draw_networkx(G_sub, pos = coords, with_labels=False, node_size=1,edge_color=colors, arrowsize = 3,
                    node_color = 'grey', width = 3, label = colors_dict) # plot graph
    
    # build legend
    leg = defaultdict(list)
    for l in all_lines:
        leg[colors_dict[l]].append(l)
        
    legend_handles = []
    for k, v in leg.items():
        legend_handles.append(mpatches.Patch(color=k, label= ', '.join(v)))
        
    plt.legend(handles=legend_handles)

    
from IPython.display import display_html
from itertools import chain,cycle

def display_side_by_side(*args,titles=cycle([''])):
    """
    Auxiliary function to print dataframes side by side.
    """
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2>{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)

# generate agents objects
def gen_agents_objects(agents, percorsi, foot, orari, G):
    agents_list = []
    couldntreach = []
    i = 0
    # for i, a in tqdm(enumerate(agents.values())):
    for k in tqdm(agents):
        a = agents[k]    
        home = a["Home"] # stop 
        acts_ = a['Activities']
        acts_.sort(key = lambda act__: act__[1])

        # extra
        nil = a["NIL"]
        age = a["Age"]
        all_activities = [x[0] for x in acts_]

        last_act_ = acts_[-1][0]
        schedule = {moment : (acts_[moment][2], acts_[moment][1]) 
                    if acts_[moment][2] != "home" else (home, acts_[moment][1]) 
                    for moment in range(len(acts_))}

        # we want our agents to "go back home" after their last activity 
        last_time = schedule[len(schedule)-1][1]

        d_last_times = {'dormire, mangiare e altra cura della persona': 120,
                       'istruzione e formazione': 330,
                       'lavoro retribuito': 440,
                       'tempo libero': 90}

        d_last_times2 = {'dormire, mangiare e altra cura della persona': 120,
                       'istruzione e formazione': 165, # half of the average duration
                       'lavoro retribuito': 220, # half of the average duration
                       'tempo libero': 90}

        # lavoro familiare has location = home
        if last_act_ != 'lavoro familiare':

            if last_time <= 780: # if last activity is in the first "half" of the day (before 1pm)
                if random.random() < 0.7:
                    back_home = last_time + 40 + d_last_times[last_act_] + random.randrange(-10,10) 
                  # t go home = t last act + 40 min (avg traveling time) + avg duration of last activity + noise
                else:
                    back_home = last_time + 40 + d_last_times2[last_act_] + random.randrange(-10,10)
                schedule[len(schedule)] = (home, back_home)
            else:
                if random.random() < 0.7:
                    critical_time = last_time + 40 + d_last_times2[last_act_]
                    if critical_time <= 1310:
                        back_home = critical_time + random.randrange(-10,10)
                        schedule[len(schedule)] = (home, back_home)

                else:
                    critical_time = last_time + 40 + d_last_times[last_act_]
                    if critical_time <= 1310:
                        back_home = critical_time + random.randrange(-10,10)
                        schedule[len(schedule)] = (home, back_home)

        try:
            path = get_a_path(home, schedule[0][0], schedule[0][1], percorsi, foot, orari, G) # from home to first activity
        except:
            if home == schedule[0][0]:
                path = []
            else:
                path = ["skip"]
            couldntreach.append(i)

        for j in range(len(schedule)-1):
            try:
                curr_path = get_a_path(schedule[j][0], schedule[j+1][0], schedule[j+1][1], percorsi, foot, orari, G)
            except:
                if schedule[j][0] == schedule[j+1][0]:
                    curr_path = []
                else:
                    curr_path = ["skip"]
                couldntreach.append(i)
            path += curr_path                 
        agents_list.append(Agent(i, schedule, path, home, nil, age, all_activities))
        i += 1
    print("Number of skipped routes:",len(couldntreach))
    return agents_list


#####################################################################################
################################### AGENTS' CLASS ###################################
#####################################################################################


class Agent():
    
    def __init__(self, unique_id, schedule, path, home, nil, age, all_activities):
        
        self.unique_id = unique_id
        self.home = home
        
        self.moment = 0
        self.schedule = schedule # dict with tuples of stops for activities and times when agent should go
        self.state = "busy"
        self.current_position = home
        self.walk_counter = None
        self.step = 0
        self.path = path # series of edges
        self.waiting_time = 0
        self.traveling_time = 0
        
        # extra
        self.age = age
        self.nil = nil
        self.all_activities = all_activities
        
    def __repr__(self):
        return f"Agent {self.unique_id} is currently at {self.current_position}"
        
    def next_edge(self):
        if self.step < len(self.path):
            return self.path[self.step]
        else:
            return None
        
    def current_destination(self):
        if self.moment < len(self.schedule):
            return self.schedule[self.moment][0]
        else:
            return None

    def count_times(self):
        if self.state == "traveling":
            self.traveling_time += 1
        elif self.state == "waiting":
            self.waiting_time += 1 
            
    def reset(self):
        self.moment = 0
        self.step = 0
        self.waiting_time = 0
        self.traveling_time = 0
        self.current_position = self.home
        self.walk_counter = None
        self.state = "busy" 



#################################################################################
################################### THE MODEL ###################################
#################################################################################


def model(G, agents_list, d_active_edges, start = 300, end = 1500):
    '''
    Simulates movement of agents from agents_list on the graph G from time start to end
    '''
   
    full_edges = defaultdict(lambda: np.zeros(end-start))
    current_positions = defaultdict(list)
    moving_agents = defaultdict(list)
    traveling_on_edge = defaultdict(int)
    traveling_on_edge_set = defaultdict(set)
    waiting_on_edge = defaultdict(int)
    daily_passengers = defaultdict(int)
    daily_waiters = defaultdict(int)

    agents_busy = [a.unique_id for a in agents_list]
    agents_traveling = []
    agents_waiting = []

    
    for t in tqdm(range(start, end)):
        #----------------------------------- loop over busy agents--------------------------------#
        busy_copy = agents_busy.copy()
        random.shuffle(busy_copy)
        
        for id_ in busy_copy:
            a = agents_list[id_]
            # if it is time to leave
            if t >= a.schedule[a.moment][1]: # next time of departure
                
                if a.current_position == a.current_destination():
                    a.moment += 1
                    if a.moment == len(a.schedule):
                        a.state = "finished"
                        agents_busy.remove(id_)

                elif a.next_edge() == "skip":
                    a.current_position = a.current_destination()
                    a.moment += 1
                    if a.moment == len(a.schedule):
                        a.state = "finished"
                        agents_busy.remove(id_)
                    else:
                        a.step += 1

                else:
                    a.state = 'waiting'
                    agents_busy.remove(id_)
                    agents_waiting.append(id_) # list of all agents waiting
                    edge = a.next_edge()
                    waiting_list(G, edge).append(id_) # list of agents waiting on edge
                    # storing info
                    waiting_on_edge[edge] = max(waiting_on_edge[edge],len(waiting_list(G, edge))) 
                    daily_waiters[edge[0]] += 1
        
        #-------------------------------- loop over traveling agents--------------------------------#
        traveling_copy = agents_traveling.copy()
        random.shuffle(traveling_copy)
        
        for id_ in traveling_copy:
            a = agents_list[id_]
            edge = a.current_position
            line_next_edges = next_edge(G, edge) # list of next edges of the line
            
            # check if the mean of transport is arrived
            arrived = False 
            if next_edge(G, edge) == []: 
                if a.walk_counter == 0:
                    arrived = True
                else:
                    a.walk_counter -= 1                            
            else:
                for e in line_next_edges: # check all the next edges of the line
                    if t%1440 in d_active_edges[e]: # if at least one of them is active...
                        arrived = True #... the check is true
                        break
                
            if not arrived:
                pass
            
            # if the means of transport the agent is on has arrived
            elif (next_edge(G, edge) == [] and G.out_degree[edge[1]] > 0 and a.current_position[2] != 'foot') or arrived:
                # agent arrives to its current destination
                if edge[1] == a.current_destination():
                    a.current_position = edge[1]
                    a.state = 'busy'
                    agents_busy.append(id_)
                    agents_traveling.remove(id_)
                    passengers_list(G, edge).remove(id_)
                    a.moment += 1
                    if a.moment == len(a.schedule): # if it was the last destination
                        a.state = "finished" 
                        agents_busy.remove(id_)
                
                # agent changes line/the next edge is not active --> gets off and waits at the stop
                elif edge[2] != a.next_edge()[2] or not t%1440 in d_active_edges[a.next_edge()]: 
                    a.current_position = edge[1]
                    a.state = 'waiting'
                    agents_traveling.remove(id_)
                    agents_waiting.append(id_)
                    passengers_list(G, edge).remove(id_)
                    
                    waiting_list(G, a.next_edge()).append(id_)
                    # storing info
                    daily_waiters[a.next_edge()[0]] += 1
                    
                # agent gets on their next edge (if it is active)
                elif edge[2] == a.next_edge()[2] and t%1440 in d_active_edges[a.next_edge()]:
                    a.current_position = a.next_edge()
                    
                    passengers_list(G, edge).remove(id_)
                    passengers_list(G, a.next_edge()).append(id_)
                    # storing info
                    traveling_on_edge[a.next_edge()] = max(traveling_on_edge[a.next_edge()],len(passengers_list(G, a.next_edge())))
                    traveling_on_edge_set[a.next_edge()].update(passengers_list(G, a.next_edge()))
                    daily_passengers[a.next_edge()] += 1
                    
                    a.step += 1
                    if next_edge(G, a.current_position) == []:
                        a.walk_counter = weight(G, a.current_position) 
        
        #---------------------------------- loop over waiting agents--------------------------------#
        waiting_copy = agents_waiting.copy()
        random.shuffle(waiting_copy)  
        
        for id_ in waiting_copy: # adjust id_/a..
            a = agents_list[id_]
            edge = a.next_edge()
            if not t%1440 in d_active_edges[edge]: #adjust
                continue

            # the agent tries to get on their next edge
            if t%1440 in d_active_edges[a.next_edge()]:
                availability = min(len(waiting_list(G, edge)), total_capacity(G, edge) - len(passengers_list(G, edge)))

                # if there's space on the vehicle it gets on
                if id_ in set(waiting_list(G, edge)[:availability]): 
                    a.current_position = edge            
                    passengers_list(G, edge).append(id_)
                    # storing info
                    traveling_on_edge[edge] = max(traveling_on_edge[edge],len(passengers_list(G, edge)))
                    traveling_on_edge_set[edge].update(passengers_list(G, edge))
                    daily_passengers[edge] += 1

                    waiting_list(G, edge).remove(id_)
                    a.state = 'traveling'
                    agents_traveling.append(id_)
                    agents_waiting.remove(id_)
                    a.step += 1
                    if next_edge(G, a.current_position) == []:
                        a.walk_counter = weight(G, a.current_position)


        for agent_ in agents_list:
            if agent_.state != 'busy' and agent_.state != 'finished':
                current_positions[t%1440].append(agent_.current_position)
            else:
                current_positions[t%1440].append(agent_.state)
            agent_.count_times()
            
        moving_agents[t%1440] = [len(agents_traveling)/ len(agents_list), len(agents_waiting)/len(agents_list)]
        
        for edge in G.edges:
            if edge[2] != "foot":
                availability = total_capacity(G, edge) - len(passengers_list(G, edge))
                if availability == 0:
                    full_edges[edge][t-start] = 1

    history_ = pd.DataFrame(current_positions)

    for i in traveling_on_edge_set.keys():
        traveling_on_edge_set[i] = len( traveling_on_edge_set[i])
        
            
    print("Simulation completed successfully!")
    return agents_list, history_, traveling_on_edge, traveling_on_edge_set, moving_agents, daily_passengers, daily_waiters, full_edges


##################################################################################
################################### SIMULATION ###################################
##################################################################################

def simulation(G, n = False, start = 300, end = 1500, import_agents = False, import_agents_ob = False, reset_agents = True,
               d_ages = d_ages, d_activities = d_activities, age_distro = age_distro, nil_age = nil_age, fermate_nils = fermate_nils, 
               activities = activities, am_counts_df_prob = am_counts_df_prob, am_counts_df = am_counts_df, rescaled_ts = rescaled_ts,
               percorsi = percorsi, foot = foot, orari = orari, d_active_edges = d_active_edges):
    '''
    Generates/ imports agents and runs the simulation. 
    Specify the number n of agents to generate. 
    To import agents object from pickle, write the path in import_agents_ob.
    To import agents' schedules from pickle, write the path in import_agents.
    
    '''

    if import_agents == False and import_agents_ob == False and n == False:
        raise Exception("The number of agents must be specified!")

    elif import_agents == False and import_agents_ob == False and n != False:
        tic = time.time()
        agents, agents_df = gen_agent(n, d_ages, d_activities)
        toc = time.time()
        print(f"Generated {n} agents in", round(toc-tic,1),"seconds")
        print("Initializing agents:")
        tic = time.time()
        agents_list = gen_agents_objects(agents, percorsi, foot, orari, G)
        toc = time.time()
        print(f"Initialized {len(agents_list)} agents objects in", round(toc-tic,1),"seconds")
    
    elif import_agents != False and import_agents_ob == False:
        file = open(import_agents, 'rb')
        agents = pickle.load(file)
        file.close()
        print("Initializing imported agents:")
        tic = time.time()
        agents_list = gen_agents_objects(agents, percorsi, foot, orari, G)
        toc = time.time()
        print(f"Initialized {len(agents_list)} agents objects in", round(toc-tic,1),"seconds")

    elif import_agents == False and import_agents_ob != False:
        file = open(import_agents_ob, 'rb')
        agents_list = pickle.load(file)
        file.close()
        print(f"Imported {len(agents_list)} agents objects")
    
    else:
        raise Exception("Cannot import both agents and agents objects!")

    if reset_agents == True:
        for a in agents_list:
            a.reset()
    
    print("Running the simulation:")
    agents_list2, history_, traveling_on_edge, traveling_on_edge_set, moving_agents, daily_passengers, daily_waiters, full_edges = model(G, agents_list, d_active_edges, start = start, end = end)

    return agents_list2, history_, traveling_on_edge, traveling_on_edge_set, moving_agents, daily_passengers, daily_waiters, full_edges

##################################################################################
################################### STATISTICS ###################################
##################################################################################

def summary_stats(agents_list, sum_print = True):
    '''
    Returns summary statistics on an executed simulation. If sum_print = True, 
    also prints.
    '''
    waiting = 0
    traveling = 0
    avg_waiting_time_tot = 0
    avg_traveling_time_tot = 0
    avg_number_activities_tot = 0
    avg_waiting_time_fin = 0
    avg_traveling_time_fin = 0
    avg_number_activities_fin = 0

    for a in agents_list:
        avg_waiting_time_tot += a.waiting_time
        avg_traveling_time_tot += a.traveling_time
        avg_number_activities_tot += len(a.schedule)
        if a.state != "traveling" and a.state != "waiting":
            avg_waiting_time_fin += a.waiting_time
            avg_traveling_time_fin += a.traveling_time
        if a.state == "waiting":
            waiting += 1
        elif a.state == "traveling":
            traveling += 1
            
    avg_waiting_time_tot = avg_waiting_time_tot / len(agents_list)
    avg_traveling_time_tot = avg_traveling_time_tot / len(agents_list)
    avg_number_activities = avg_number_activities_tot / len(agents_list)
    avg_waiting_time_fin = avg_waiting_time_fin / (len(agents_list) - traveling - waiting)
    avg_traveling_time_fin = avg_traveling_time_fin / (len(agents_list) - traveling - waiting)

    if sum_print == True:
        print("Total number of agents:", len(agents_list))
        print("Number of agents still waiting:", waiting)
        print("Number of agents still traveling:", traveling)
        print("Average waiting time:", "all agents:", round(avg_waiting_time_tot, 1), "only finished agents:", round(avg_waiting_time_fin, 1))
        print("Average traveling time:","all agents:", round(avg_traveling_time_tot, 1), "only finished agents:", round(avg_traveling_time_fin, 1))
        print("Average number of activities:", round(avg_number_activities,1))
        print("Average traveling time per destination:","all agents:",round(avg_traveling_time_tot/avg_number_activities, 1),"only finished agents:",round(avg_traveling_time_fin/avg_number_activities, 1))
        print("Average waiting time per destination:","all agents:",round(avg_waiting_time_tot/avg_number_activities, 1),"only finished agents:",round(avg_waiting_time_fin/avg_number_activities, 1))
    
    return avg_waiting_time_tot, avg_traveling_time_tot, avg_waiting_time_fin, avg_traveling_time_fin, avg_number_activities


def moving_ts_plot(moving_agents):
    '''
    Plots the percentage of traveling and waiting agents over time.
    '''
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

    ts = [minutes_to_str(x) for x in moving_agents.keys()]
    title = ["% Traveling", "% Waiting"]

    for i in range(2):
        axs[i].plot(ts, [x[i] for x in moving_agents.values()])
        axs[i].set_xticks(ticks = range(0,1200,120))
        axs[i].set_title(title[i], size = 15)

    plt.show()
    
def agg_moving_ts_plot(moving_agents):
    '''
    Plots the percentage of traveling and waiting agents over time averaged across different simulations.
    '''
    traveling = pd.DataFrame(moving_agents[0]).T[0]
    waiting = pd.DataFrame(moving_agents[0]).T[1]
    for i in range(1,len(moving_agents)):
        traveling = pd.concat([traveling, pd.DataFrame(moving_agents[i]).T[0] ], axis = 1)
        waiting = pd.concat([waiting, pd.DataFrame(moving_agents[i]).T[1] ], axis = 1)
    traveling_mean = traveling.mean(axis = 1)
    waiting_mean = waiting.mean(axis = 1)
    traveling_std = traveling.std(axis = 1)
    waiting_std = waiting.std(axis = 1)
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

    ts = [minutes_to_str(x) for x in moving_agents[0].keys()]
    title = ["% Traveling", "% Waiting"]
    
    axs[0].plot(ts, traveling_mean, c = "darkred")
    axs[0].set_xticks(ticks = range(0,1200,120))
    axs[0].set_title(title[0], size = 15)
    axs[0].fill_between(ts, traveling_mean+1.96*traveling_std, traveling_mean-1.96*traveling_std, facecolor=(0.805, 0.34, 0.27), alpha=0.5)
    axs[0].set_ylim([0.0,0.14]) 
    
        
    axs[1].plot(ts, waiting_mean, c = "darkred")
    axs[1].set_xticks(ticks = range(0,1200,120))
    axs[1].set_title(title[1], size = 15)
    axs[1].fill_between(ts,  waiting_mean+1.96*waiting_std,  waiting_mean-1.96*waiting_std, facecolor=(0.805, 0.34, 0.27), alpha=0.5)
    axs[1].set_ylim([0.0,0.14])
    plt.show()

def density_graph_plot(traveling_on_edge, traveling_on_edge_set, daily_waiters, G, coords = coords):
    '''
    Plot the network after simulation with:
    - the size and color intensity of nodes proportional to the total number of agents that have
    passed by each of them; 
    - the width of edges proportional to the total number of agents that have passed by each of them;
    - the color intensity of edges proportional to the maximum number of agents simultaneously traveling on each of them. 
    '''
    colors = {1: 'red', 2: 'green', 3: 'yellow', 4: 'blue', 5: 'violet', 'TRAM': 'brown', 'BUS': 'pink', 'FILOBUS': 'grey'}
    # highlights traffic/density on edges
    max_density = max(list(traveling_on_edge.values()))
    width = [1 + ((6 - 1) / (max_density - 1)) * (i - 1) for i in traveling_on_edge.values()] # rescaling edge widths from 1 to 6
    high_density_c = np.quantile(list(traveling_on_edge_set.values()), 0.95)

    nodes_sizes = [daily_waiters[i] if i in daily_waiters else 0 for i in list(G)]
    maxi = max(nodes_sizes)
    mini = min(nodes_sizes)
    node_size = [1 + ((50 - 1) / (maxi - mini)) * (i - mini) for i in nodes_sizes] # rescaling nodes size from 1 to 50
    high_density_n = np.quantile(nodes_sizes, 0.95)

    plt.figure(figsize=(20,20), dpi=500)
    color_comp_e = [cm.Reds(traveling/high_density_c) for traveling in traveling_on_edge_set.values()]
    color_comp_n = [cm.Blues(waiting/high_density_n) for waiting in nodes_sizes]
    nx.draw_networkx(G, pos = coords, with_labels=False, edgelist= list(traveling_on_edge.keys()), node_size=node_size,
                    edge_color=color_comp_e, node_color = color_comp_n, edge_cmap = "Reds" , cmap = "Blues", 
                     arrowsize = 0.5, width = width, label = colors)
    plt.show()  


def n_hotspots(n, daily_waiters, fermate = fermate):
    '''
    Returns dataframe of top n stops by total number of agents that have passed by there.
    '''
    return pd.DataFrame([daily_waiters]).T.sort_values(0, ascending = False).head(n).reset_index().merge(fermate, 
                        right_on = 'id_amat', left_on='index')[['id_amat', 'nome', 0]].rename(columns = {0: 'daily_waiters'})

def n_hotedges(n, daily_passengers, fermate = fermate):
    '''
    Returns dataframe of top n edges by total number of agents that have passed by there.
    '''
    df = pd.DataFrame([daily_passengers]).T.sort_values(0, ascending = False).head(n).reset_index()
    df['from'] = df['index'].apply(lambda x: x[0])
    df['to'] = df['index'].apply(lambda x: x[1])
    df['line'] = df['index'].apply(lambda x: x[2])
    df = df.merge(fermate, right_on = 'id_amat', left_on='from').merge(fermate, right_on = 'id_amat', left_on='to')
    return df[['from', 'nome_x', 'to', 'nome_y', 'line', 0]].rename(columns = {'nome_x': 'from_name', 'nome_y': 'to_name', 0: 'daily_passengers'})


def full_edges_df(full_edges):
    """
    Returns a df sorted by total number of minutes each edge reached full capacity.
    """
    df = pd.DataFrame([full_edges]).T
    df["Full Minutes"] = df[0].apply(lambda x: len(x[x==1]))
    
    return df[["Full Minutes"]].sort_values("Full Minutes", ascending = False)


def vehicle_changes(agent_list, actual = True):
    """
    Computing the average number of vehicle changes for a given list of agent objects.
    """
    counters = []
    for i in agent_list:
        
        if len(i.path) != 0:
            
            if actual == True:
                path = i.path[:i.step+1]
            else:
                path = i.path 
                
            counter = int(path[0][2] != "foot")
            for j in range(len(path) - 1):
                if path[j][2] != path[j+1][2] and path[j+1][2] != "foot":
                    counter +=1
            counters.append(counter)
        
    return np.mean(counters)


def different_vehicles(agent_list, actual = True):
    """
    Returning the average number of different means used over the course of the simulation and a dictionary with the set of
    means for each agent.
    """
    avg_means = []
    d = {}
    for a in agent_list:
            
        if len(a.path) != 0:
            
            if actual == True:
                path = a.path[:a.step+1]
            else:
                path = a.path
                
            means = set()
            
            for edge in path:
                means.add(edge[2])

            d[a.unique_id] = means    
            avg_means.append(len(means))
        
    return np.mean(avg_means), d 

def vehicle_usage(agent_list, actual = True):
    """
    Returns total number of passengers for each vehicle.
    """
    d = defaultdict(int)
    for a in agent_list:
        if len(a.path) != 0:
            if actual == True:
                path = a.path[:a.step+1]
            else:
                path = a.path
            for edge in path:
                d[edge[2]] += 1
            
    df = pd.DataFrame([d]).T.rename(columns = {0: "Total Passengers"}).sort_values("Total Passengers",ascending = False)
    
    return df.drop("foot", axis = 0)     


def foot_route(agent_list, actual = True):
    """
    Returns average number of times agents move on foot.
    """
    tot_feet = []
    for a in agent_list:
        if len(a.path) != 0:
            
            if actual == True:
                path = a.path[:a.step+1]
            else:
                path = a.path
                
            for edge in path:
                feet = int(path[0][2] == "foot")
                for j in range(len(path) - 1):
                    if path[j+1][2] == "foot" and path[j][2] != "foot":
                        feet +=1
            tot_feet.append(feet)
    
    return np.mean(tot_feet)   


def stops_by_vehicle(agent_list, actual = True):
    """
    Returns average number of stops traveled on each vehicle.
    """
    d = defaultdict(list)
    for a in agent_list:
        if len(a.path) != 0:
            
            if actual == True:
                path = a.path[:a.step+1]
            else:
                path = a.path
                
            counter = 0
            current = path[0][2]
            for edge in path:
                if current == edge[2]:
                    counter += 1
                else:
                    d[current].append(counter) 
                    current = edge[2]
                    counter = 1
                    
            d[current].append(counter)
    df = pd.DataFrame([d]).T
    
    df["Average Stops"] = df[0].apply(lambda x: np.mean(x)) 
    df["Median Stops"] = df[0].apply(lambda x: np.quantile(x, 0.5))
    df["Max Stops"] = df[0].apply(lambda x: max(x))
    df["Min Stops"] = df[0].apply(lambda x: min(x))
    return df.iloc[:,1:]


def stops_in_path(agents_list, stops, actual = True):
    """
    Returns a subset of agents that either stop or change vehicle at one of the given nodes.
    """
    stops = set(stops)
    agents = set()
      
    for a in agents_list:
        
        # if given stops are in the agent's schedule
        if len(stops.intersection(set([t[0] for t in a.schedule.values()]))) > 0:
            agents.add(a.unique_id)
        
        elif len(a.path) != 0:
            if actual == True:
                path = a.path[:a.step+1]
            else:
                path = a.path            
            
            for edge in path:
                for j in range(len(path) - 1):
                    # if the agent changes vechicle at ones of the given stops
                    if path[j][2] != path[j+1][2] and path[j][1] in stops:
                        agents.add(a.unique_id)
                        break
                break
                        
    return agents

def time_distance_distribution(history, G):
    """
    Returns a list of distance-adjusted time for single destination
    """
    speeds = {1: 32, 2: 32, 3: 32, 4: 32, 5: 32, 'TRAM': 11, 'BUS': 14, 'FILOBUS': 14, 'foot': 5}
    
    distances = []   
    times = []
    time_distances = []
    
    for i in history.index:
        row = history.loc[i,:]
        trip_time = 0
        trip_distance = 0
        
        started = False
        previous_position = 0
        
        for position in row:
            if position != 'busy' and position != 'finished':
                trip_time += 1
                started = True
                if type(position) == tuple:
                    if not(previous_position == position):
                        weight = G.edges[position]['weight']
                        speed = speeds[G.edges[position]['trans_mode']]
                        trip_distance += weight*speed
                    
            elif started:
                time_distances.append(trip_time/trip_distance)
                distances.append(trip_distance)
                times.append(trip_time)
                trip_time = 0
                trip_distance = 0
                started = False
            
            if type(position) == np.int64:
                position = int(position)
            previous_position = position
            
    time_distances = np.array(time_distances)[np.array(time_distances) < np.quantile(time_distances, 0.95)]
                
    return time_distances, distances, times
