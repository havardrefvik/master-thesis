import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pandapower as pp
import glob
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import pydot
import csv
import datetime
import math

folder_name = 'dataset/10SM'
file_type = 'csv'
separator =';'

list_of_dfs = []
dict_of_dfs = {}

for f in glob.glob(folder_name + "/*."+file_type):
    list_of_dfs.append(pd.read_csv(f, sep=separator))

#dataframe = pd.concat([pd.read_csv(f, sep=seperator) for f in glob.glob(folder_name + "/*."+file_type)],ignore_index=True)

#print(list_of_dfs[0])
number_of_smart_meters = len(list_of_dfs)

#Make all the dataframes start at the same time, and go for 1000 values after that
for i in range(len(list_of_dfs)):
    list_of_dfs[i]["date_tz"] = pd.to_datetime(list_of_dfs[i]["date_tz"], utc=True)
    list_of_dfs[i] = list_of_dfs[i].set_index("date_tz")
    index = list_of_dfs[i].index.get_loc("2018-09-03 01:00:00+02:00")
    list_of_dfs[i] = list_of_dfs[i][index:index+8660]
    #print(list_of_dfs[i].head())


#Create a dictionary of dataframes from the list of dataframes
#This only collects the last value of the "Power" value in each dataframe
"""for i in range(0,number_of_smart_meters):
    #dict_of_dfs[i] = list_of_dfs[i]["kWh/h"][list_of_dfs[i].index[i]]
    dict_of_dfs[i] = list_of_dfs[i]["kWh/h"][0]"""

#print(list_of_dfs[0].head())
#print(dict_of_dfs)

# Set seed to get reproducible graph
seed = 10
random.seed(seed)
np.random.seed(seed)



# Create graph
def create_graph(number_of_nodes):
    # Create a random tree with x nodes
    global G
    G = nx.random_tree(number_of_nodes, seed=seed)
    pos = nx.nx_pydot.graphviz_layout(G, prog='dot', root=0)

    # Draw the graph
    nx.draw(G, pos, with_labels=True)
    plt.show()
    return G


create_graph(number_of_smart_meters)

#Set node attributes to the dictionary values. Each node will have the value of Power and Timestamp
#nx.set_node_attributes(G, dict_of_dfs, "Power value")

#print("Node values " + str(G.nodes(data="Power value")))

def capacity_calculation(graph, root_node, actual_node):
    # 1 dfs-tree and node levels
    # dictionary to store the "depth" or level or each node
    levels = {0: 0}
    # directed dfs tree to store the predecessors
    dfs_tree = nx.DiGraph()
    for u, v in nx.dfs_edges(G, 0):
        levels[v] = levels[u] + 1
        dfs_tree.add_edge(u, v)

    #print(levels)
    #{0: 0, 17: 1, 6: 1, 14: 2, 15: 3, 12: 4, 16: 4, 1: 5, 4: 6, 5: 6, 8: 7, 18: 8, 3: 9, 13: 9, 9: 10, 10: 4, 2: 5, 7: 6, 11: 7, 19: 8}

    # 2 sorting nodes
    # process nodes by highest level first
    ordered_nodes = sorted(G.nodes, key=lambda node: levels[node], reverse=True)

    # 3 iteratively calculation
    # final dict
    edges_succeeeding = {}
    for node in ordered_nodes:
        if G.degree(node) == 1:
            # add lowest entry to the dict
            edges_succeeeding[node] = 0

        # start adding information about the predecessor
        for predecessor in dfs_tree.predecessors(node):
            edges_succeeeding[predecessor] = edges_succeeeding.get(predecessor, 0) + edges_succeeeding[node] + 1

    #print("Edges succeeding " + str(edges_succeeeding))
    #{9: 0, 13: 1, 3: 0, 18: 3, 8: 4, 19: 0, 11: 1, 5: 5, 7: 2, 4: 0, 1: 7, 2: 3, 16: 8, 10: 4, 15: 15, 12: 0, 14: 16, 6: 17, 0: 19, 17: 0}
    standard_power_size = 11
    I = ((edges_succeeeding[actual_node] * standard_power_size) / ((math.sqrt(3) * 400 * 0.9)))

    if I < 0.091:
        return "FG7OR_16"
    elif I < 0.117:
        return "FG7OR_25"
    elif I < 0.143:
        return "FG7OR_35"
    elif I < 0.178:
        return "FG7OR_50"
    elif I < 0.221:
        return "FG7OR_70"
    elif I > 0.221:
        return "FG7OR_95"

#capacity_calculation(G, 0)

iteration = 0

def create_power_network(number_of_smart_meters, iteration):
    #Create power network
    #Had to set seed again here to make it work for line lenghts
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    # Create a dictionary of dataframes from the list of dataframes
    for i in range(0, number_of_smart_meters):
        # dict_of_dfs[i] = list_of_dfs[i]["kWh/h"][list_of_dfs[i].index[i]]
        dict_of_dfs[i] = list_of_dfs[i]["kWh/h"][iteration]

    # Set node attributes to the dictionary values. Each node will have the value of Power and Timestamp
    nx.set_node_attributes(G, dict_of_dfs, "Power value")
    print("Node values " + str(G.nodes(data="Power value")))

    net = pp.create_empty_network()

    # Linedata
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 1.540,
                 'x_ohm_per_km': 0.075, 'max_i_ka': 0.091,
                 'type': 'cs'}
    pp.create_std_type(net, line_data, name='FG7OR_16', element='line')

    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.990,
                 'x_ohm_per_km': 0.074, 'max_i_ka': 0.117,
                 'type': 'cs'}
    pp.create_std_type(net, line_data, name='FG7OR_25', element='line')

    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.710,
                 'x_ohm_per_km': 0.072, 'max_i_ka': 0.143,
                 'type': 'cs'}
    pp.create_std_type(net, line_data, name='FG7OR_35', element='line')

    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.500,
                 'x_ohm_per_km': 0.071, 'max_i_ka': 0.178,
                 'type': 'cs'}

    pp.create_std_type(net, line_data, name='FG7OR_50', element='line')

    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.350,
                 'x_ohm_per_km': 0.070, 'max_i_ka': 0.221,
                 'type': 'cs'}
    pp.create_std_type(net, line_data, name='FG7OR_70', element='line')

    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.270,
                 'x_ohm_per_km': 0.069, 'max_i_ka': 0.266,
                 'type': 'cs'}
    pp.create_std_type(net, line_data, name='FG7OR_95', element='line')

    #Create buses
    #Create grid connection
    bus0 = pp.create_bus(net, vn_kv=20., name="Bus 0", index=0)

    dict_of_buses = {}

    for i in range(1, number_of_smart_meters):
        #bus = pp.create_bus(net, vn_kv=0.4, name="Bus " + str(i), index=i)
        dict_of_buses[i] = pp.create_bus(net, vn_kv=0.4, name="Bus " + str(i), index=i)
        #print(net.load)


    #Create bus elements
    dict_of_buses[0] = pp.create_ext_grid(net, bus=bus0, vm_pu=1.02, name="Grid Connection")

    #Creating loads




    # create branch elements
    #trafo = pp.create_transformer(net, hv_bus=bus0, lv_bus=dict_of_buses[1], std_type="0.4 MVA 20/0.4 kV", name="Trafo")

    #Create dictionary of lines to make sure it is only one line between buses
    dict_of_lines = {}

    # Test to create integrated
    pp.create_bus(net, vn_kv=0.4, name="Bus 10'", index=10)
    pp.create_bus(net, vn_kv=0.4, name="Bus 11'", index=11)
    pp.create_bus(net, vn_kv=0.4, name="Bus 12'", index=12)
    pp.create_bus(net, vn_kv=0.4, name="Bus 13'", index=13)
    pp.create_bus(net, vn_kv=0.4, name="Bus 14'", index=14)
    pp.create_bus(net, vn_kv=0.4, name="Bus 15'", index=15)
    pp.create_bus(net, vn_kv=0.4, name="Bus 16'", index=16)
    pp.create_bus(net, vn_kv=0.4, name="Bus 17'", index=17)
    pp.create_bus(net, vn_kv=0.4, name="Bus 18'", index=18)
    pp.create_bus(net, vn_kv=0.4, name="Bus 19'", index=19)

    for i in range(1, number_of_smart_meters):
        pp.create_load(net, bus=dict_of_buses[i]+10, p_mw=G.nodes[i]["Power value"]/1000, index=i+10)
        print("Bus " + str(i+10) + " power value " + str(G.nodes[i]["Power value"]))

    #Creating lines
    for i in range(len(dict_of_buses)):
            for j in range(len(list(G.neighbors(i)))):
                if list(G.neighbors(i))[j] not in dict_of_lines: #If not in dictionary, we add the line to the dictionary
                    #print(G.neighbors(i))
                    line = pp.create_line(net, from_bus=dict_of_buses[i], to_bus=list(G.neighbors(i))[j], length_km=0.04*random.randint(1,10), std_type=capacity_calculation(G, 0, i), \
                                          name="Line from bus " + str(dict_of_buses[i]) + " to bus " + str(list(G.neighbors(i))[j]))

                    dict_of_lines[dict_of_buses[i]] = list(G.neighbors(i))[j]

    line = pp.create_line(net, from_bus=0, to_bus=10, length_km=0.04*random.randint(1,10), std_type="FG7OR_50", name="Line from bus 0 to 10")
    line = pp.create_line(net, from_bus=1, to_bus=11, length_km=0.04*random.randint(1,10), std_type="FG7OR_50", name="Line from bus 1 to 11")
    line = pp.create_line(net, from_bus=2, to_bus=12, length_km=0.04 * random.randint(1, 10), std_type="FG7OR_50",
                          name="Line from bus 2 to 12")
    line = pp.create_line(net, from_bus=3, to_bus=13, length_km=0.04 * random.randint(1, 10), std_type="FG7OR_50",
                          name="Line from bus 3 to 13")
    line = pp.create_line(net, from_bus=4, to_bus=14, length_km=0.04 * random.randint(1, 10), std_type="FG7OR_50",
                          name="Line from bus 4 to 14")
    line = pp.create_line(net, from_bus=5, to_bus=15, length_km=0.04 * random.randint(1, 10), std_type="FG7OR_50",
                          name="Line from bus 5 to 15")
    line = pp.create_line(net, from_bus=6, to_bus=16, length_km=0.04 * random.randint(1, 10), std_type="FG7OR_50",
                          name="Line from bus 6 to 16")
    line = pp.create_line(net, from_bus=7, to_bus=17, length_km=0.04 * random.randint(1, 10), std_type="FG7OR_50",
                          name="Line from bus 7 to 17")
    line = pp.create_line(net, from_bus=8, to_bus=18, length_km=0.04 * random.randint(1, 10), std_type="FG7OR_50",
                          name="Line from bus 8 to 18")
    line = pp.create_line(net, from_bus=9, to_bus=19, length_km=0.04 * random.randint(1, 10), std_type="FG7OR_50",
                          name="Line from bus 9 to 19")


    #print(net.trafo)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(net.line)

    #print(net.trafo)
    #print(net.bus)
    print(net.load)

    # Run power flow
    pp.runpp(net)
    #print("POWER FLOW RUNNING")
    print(net.res_bus["vm_pu"])
    print("ITERATION NUMBER " + str(iteration))

    #Writing to file to get new time series
    """f = open('new_results_voltage.csv', 'a')

    with f:
        writer = csv.writer(f)
        writer.writerow(net.res_bus["vm_pu"])"""
    print(net.res_bus["p_mw"])


#For loop to do it x amount of times (for making a csv file with time series)
"""for i in range(0, 8660):
    create_power_network(number_of_smart_meters, iteration)
    iteration += 1"""

for i in range(0, 1000):
    create_power_network(number_of_smart_meters, iteration)
    iteration += 1

#Regular function call
#create_power_network(number_of_smart_meters, 0)