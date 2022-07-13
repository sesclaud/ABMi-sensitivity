import pandas as pd
from killing_agents import full_edges_df, Agent
import copy

def write_agents(agents_list, file="./agents_list.csv"):
    with open(file,"w",encoding="utf-8") as f:
        f.write("unique_id;home;moment;schedule;state;current_position;walk_counter;step;path;waiting_time;traveling_time;age;nil;all_activities")
        for a in agents_list:
            f.write(f"\n{a.unique_id};\
                    {a.home};\
                    {a.moment};\
                    {str(a.schedule)};\
                    {a.state};\
                    {a.current_position};\
                    {a.walk_counter};\
                    {a.step};\
                    {str(a.path)};\
                    {a.waiting_time};\
                    {a.traveling_time};\
                    {a.age};\
                    {a.nil};\
                    {str(a.all_activities)}".replace("                    ","")) # need to remove the long spaces that are added with the multiline string

def parse_edge(ed):
    o = ed.replace("(","").replace(")","").replace("'","").split(", ")
    # print(o)
    if "skip" in o:
        return o
    else:
        return (int(o[0]),int(o[1]),o[2])

def read_agents(file="./agents_list.csv"):

    def read_schedule(sched):
        d = {}
        for i in sched.replace("{","").replace(")}","").split("), "):
            o = i.split(": (")
            k = int(o[0])
            v = tuple(int(x) for x in o[1].split(", "))
            d[k] = v
        return d

    def read_path(path):
        l = []
        if path != "[]":
            for i in path.replace("[","").replace("]","").split("), ("):
                l.append(parse_edge(i))
        # else:
        #     print("path empty")
        return l

    def read_all_act(aa):
        return [i.replace("'","") for i in aa.replace("[","").replace("]","").split(", '")]

    agents_list = []
    with open(file,"r",encoding="utf-8") as f:
        lines = f.readlines()
    ct = 0
    for row in lines:
        ct += 1
        # print(ct)
        if "unique_id" not in row:
            row_items = row.replace("\n","").split(";")
            a = Agent(0,0,[],0,0,0,[])
            a.unique_id = int(row_items[0])
            a.home = int(row_items[1])
            a.moment = int(row_items[2])
            a.schedule = read_schedule(row_items[3])
            a.state = row_items[4] # busy
            a.current_position  = int(row_items[5]) if row_items[5].isdigit() else parse_edge(row_items[5])
            a.walk_counter = int(row_items[6]) if row_items[6] != "None" else None
            a.step = int(row_items[7])
            a.path = read_path(row_items[8])
            a.waiting_time = int(row_items[9]) 
            a.traveling_time = int(row_items[10])
            a.age = row_items[11]
            a.nil = row_items[12]
            a.all_activities = read_all_act(row_items[13])
            agents_list.append(a)

    return agents_list

def save_experiment(sim_output, path="./"):
    print("Saving 8 objects... ",end="")
    agents_list, history_, traveling_on_edge, traveling_on_edge_set, moving_agents, daily_passengers, daily_waiters, full_edges = sim_output
    print("1, ", end="")
    write_agents(agents_list,file=path+"exp_agents_list.csv")
    print("2, ", end="")
    stringify = lambda x: str(x)
    history_2 = copy.deepcopy(history_)
    history_2.columns = [str(i) for i in history_2.columns]
    for c in history_2.columns:
        history_2[c] = history_2[c].apply(stringify)
    history_2.to_parquet(path+"exp_history.parquet")
    print("3, ", end="")
    pd.DataFrame.from_dict(traveling_on_edge,orient="index").to_csv(path+"exp_traveling_on_edge.csv",encoding="utf-8")
    print("4, ", end="")
    pd.DataFrame.from_dict(traveling_on_edge_set,orient="index").to_csv(path+"exp_traveling_on_edge_set.csv",encoding="utf-8")
    print("5, ", end="")
    pd.DataFrame.from_dict(moving_agents,orient="index").to_csv(path+"exp_moving_agents.csv",encoding="utf-8")
    print("6, ", end="")
    pd.DataFrame.from_dict(daily_passengers,orient="index").to_csv(path+"exp_daily_passengers.csv",encoding="utf-8")
    print("7, ", end="")
    pd.DataFrame.from_dict(daily_waiters,orient="index").to_csv(path+"exp_daily_waiters.csv",encoding="utf-8")
    print("8 ", end="")
    full_edges_df(full_edges).to_csv(path+"exp_full_edges.csv",encoding="utf-8")
    print("Experiment saved!")

def load_experiment(path="./"):
    print("Loading 8 objects... ",end="")
    print("1, ", end="")
    agents_list = read_agents(file=path+"exp_agents_list.csv")
    print("2, ", end="")
    history_ = pd.read_parquet(path+"exp_history.parquet")
    def edgify(x):
        if x.isdigit():
            return int(x)
        elif x[0] == '(':
            return parse_edge(x)
        else:
            return x
    for c in history_.columns:
        history_[c] = history_[c].apply(edgify)
    print("3, ", end="")
    traveling_on_edge = pd.read_csv(path+"exp_traveling_on_edge.csv",encoding="utf-8")
    traveling_on_edge = {edgify(traveling_on_edge['Unnamed: 0'][i]): traveling_on_edge['0'][i] for i in traveling_on_edge.index}
    print("4, ", end="")
    traveling_on_edge_set = pd.read_csv(path+"exp_traveling_on_edge_set.csv",encoding="utf-8")
    traveling_on_edge_set = {edgify(traveling_on_edge_set['Unnamed: 0'][i]): traveling_on_edge_set['0'][i] for i in traveling_on_edge_set.index}
    print("5, ", end="")
    moving_agents = pd.read_csv(path+"exp_moving_agents.csv",encoding="utf-8").set_index('Unnamed: 0')
    print("6, ", end="")
    daily_passengers = pd.read_csv(path+"exp_daily_passengers.csv",encoding="utf-8")
    daily_passengers = {edgify(daily_passengers['Unnamed: 0'][i]): daily_passengers['0'][i] for i in daily_passengers.index}
    print("7, ", end="")
    daily_waiters = pd.read_csv(path+"exp_daily_waiters.csv",encoding="utf-8")
    daily_waiters = {daily_waiters['Unnamed: 0'][i]: daily_waiters['0'][i] for i in daily_waiters.index}
    print("8 ", end="")
    full_edges = pd.read_csv(path+"exp_full_edges.csv",encoding="utf-8")
    full_edges = {edgify(full_edges['Unnamed: 0'][i]): full_edges['Full Minutes'][i] for i in full_edges.index}

    print("Experiment loaded!")

    return agents_list, history_, traveling_on_edge, traveling_on_edge_set, moving_agents, daily_passengers, daily_waiters, full_edges
    