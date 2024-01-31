# %%
import pandas as pd
import numpy as np
import scipy.io as sio
import json


# Import file
file = pd.read_csv(r"116_artery_model.txt", sep='\t')

# Convert to array and change indices to run from 0 (because Python)
artery_file = np.asarray(file)
artery_nodes = artery_file[:,0:3] - 1
artery_nodes = artery_nodes.astype(int)

# Construct the BC dictionary
def bc_dict_construct(x, y, z):
    new_dic = {}
    if y == "FLOW": bc_vals = {"Q": z[0], "t": z[1]}
    elif y == "RESISTANCE": bc_vals = {"R": z[0], "Pd": z[1]} # add "t" if necessary
    elif y == "RCR": bc_vals = {"Rp": z[0], "C": z[1], "Rd": z[2], "Pd": z[3]}
        
    values = [{"bc_name": x}, {"bc_type": y}, {"bc_values": bc_vals}]
    for val in values:
        new_dic.update(val)
    return new_dic

# Construct the junction dictionary
def j_dict_construct(x, y, z, t, w=0):
    new_dic = {}
    values = [{"inlet_vessels": x}, {"junction_name": y}, {"junction_type":z}, {"outlet_vessels":t}]
    
    for val in values:
        new_dic.update(val)
        if z == "BloodVesselJunction":
            zerod_vals = {"C": w[0], "L": w[1], "R_poiseuille": w[2], "stenosis_coefficient": w[3]}
            ves_dict = {"zero_d_element_values": zerod_vals}
            new_dic = {**ves_dict, **new_dic}
        
    return new_dic

# Construct the blood vessel dictionary
def v_dict_construct(x, y, z, t, w, bc=[]):
    new_dic = {}
    zerod_vals = {"C": w[0], "L": w[1], "R_poiseuille": w[2], "stenosis_coefficient": w[3]}
    values = [{"vessel_id": x}, {"vessel_length": y}, {"vessel_name":z}, 
              {"zero_d_element_type":t}, {"zero_d_element_values": zerod_vals}]
    
    for val in values:
        new_dic.update(val)
        if np.shape(bc)[0] > 1:
            bc_dict = {"boundary_conditions": {bc[0]:bc[1]}}
            new_dic = {**bc_dict, **new_dic}
        
    return new_dic

#######################
# BOUNDARY CONDITIONS #
#######################

b_vals = []
count=0

# AORTIC INFLOW
load_file = '/Users/chloe/Documents/Stanford/reproduce_charlton/reproduce_baseline/inflow/inflow.mat'
f = sio.loadmat(load_file)
t = f['t'].tolist()
v = f['v'].tolist()
b_vals.append( ("INFLOW"+str(count), "FLOW", [v, t] ) )

# Append all values to dictionary
#b_dict = {}
#b_list = []
#for xx in b_vals:
#    add_dict = bc_dict_construct(*xx)
#    b_dict.update(add_dict)
#    b_list.append(b_dict.copy())
    
# TERMINAL WINDKESSEL BC AT VASCULAR BED
load_file = '/Users/chloe/Documents/Stanford/reproduce_charlton/reproduce_baseline/out_wk/out_wk.mat'
f = sio.loadmat(load_file)
wk_c = f['wk_c']
wk_r = f['wk_r']
wk_p = 10 * 1333 # 10 mmHg to dynes/cm^2 (from paper)

wk_idxs = np.argwhere(~np.isnan(wk_c))
wk_c = wk_c[~(np.isnan(wk_c))]
wk_r = wk_r[~(np.isnan(wk_r))]

count = 0
for i in np.arange(0,len(wk_c)):
    b_vals.append( ("RCR"+str(count), "RCR", [ 0.09*wk_r[i], wk_c[i], 0.91*wk_r[i], wk_p] ) )
    count += 1

# Append to dictionary
b_dict = {}
b_list = []
for xx in b_vals:
    add_dict = bc_dict_construct(*xx)
    b_dict.update(add_dict)
    b_list.append(b_dict.copy())

#############
# JUNCTIONS #
#############

# Parse through data to find SPLITTING junctions with list comprehension + slicing
res = [idx for idx, val in enumerate(artery_nodes[:,1]) if val in artery_nodes[:idx,1]]
count = 0
j_vals = []
for i in np.arange(0,np.shape(res)[0],1):
    # find idx of inlet vessel
    a_idx = np.array([idx for idx, val in enumerate(artery_nodes[:,2]) if val == artery_nodes[res[i],1]])
    # find all idxs of outlet vessels
    j_idx = np.array([idx for idx, val in  enumerate(artery_nodes[:,1]) if val == artery_nodes[res[i],1]])
    # append to array of junction values
    j_vals.append( (artery_nodes[a_idx,0].tolist(), "J"+str(count), "NORMAL_JUNCTION", artery_nodes[j_idx,0].tolist()) )
    count += 1

# Parse through data to find MERGING junctions with list comprehension + slicing
res2 = [idx for idx, val in enumerate(artery_nodes[:,2]) if val in artery_nodes[:idx,2]]
for i in np.arange(0,np.shape(res2)[0],1):
    # find idx of inlet vessel
    a_idx = np.array([idx for idx, val in enumerate(artery_nodes[:,2]) if val == artery_nodes[res2[i],2]])
    # find all idxs of outlet vessels
    j_idx = np.array([idx for idx, val in enumerate(artery_nodes[:,1]) if val == artery_nodes[res2[i],2]])
    # append to array of junction values
    j_vals.append( (artery_nodes[j_idx,0].tolist(), "J"+str(count), "NORMAL_JUNCTION", artery_nodes[a_idx,0].tolist()) )
    count += 1


# Append all values to dictionary
j_dict = {}
j_list = []
for xx in j_vals:
    add_dict = j_dict_construct(*xx)
    j_dict.update(add_dict)
    j_list.append(j_dict.copy())
    
# print what json file would look like
# obj = {"junction": j_list}
# print(json.dumps(obj,indent=4))

#############
#  VESSELS  #
#############

# properties of blood
rho = 1.06
E   = 300*1000 # elastic modulus (300kPa) converted to dynes/cm^2, CHECK????
mu  = 0.04

# Initialize vessel array
v_vals = []
count = 0
idx = 0
v_list = []
for i in np.arange(0,np.shape(artery_file)[0],1):
    
    v_vals = []
    v_dict = {}
    
    # append to array of junction values
    r = np.mean([artery_file[i,6],artery_file[i,9]]) * 10**-3 # radius of vessel
    h = 0.1 * r # thickness of blood vessel
    l = artery_file[i,3] * 10**-3
    C = 3*l*np.pi*r**3/(2*E*h)
    L = rho*l/(np.pi*r**2)
    R_poiseuille = 8*mu*l/(np.pi*r**4)
    stenosis_coefficient = 0.0
    
    if i == 0:
        v_vals.append( (count, l, artery_file[i,-1], "BloodVessel", [C, L,  R_poiseuille, stenosis_coefficient], ["inlet", "INFLOW0"] ) )
    elif i in wk_idxs[:,1]:
        v_vals.append( (count, l, artery_file[i,-1], "BloodVessel", [C, L,  R_poiseuille, stenosis_coefficient], ["outlet", "RCR"+str(idx)] ) )
        idx += 1
    else:
        v_vals.append( (count, l, artery_file[i,-1], "BloodVessel", [C, L,  R_poiseuille, stenosis_coefficient]) )
    
    for xx in v_vals:
        add_dict = v_dict_construct(*xx)
    v_dict.update(add_dict)
    v_list.append(v_dict.copy())
    
    count += 1
    

#####################
# SIMULATION PARAMS #
#####################
num_cardiac_cycles  = 10
num_pts_per_cardiac = 5

sim_params = {
        "number_of_cardiac_cycles": num_cardiac_cycles,
        "number_of_time_pts_per_cardiac_cycle": num_pts_per_cardiac
}

#####################
#   CALIBRATION     #
#####################
tol_grad = 1e-05
tol_inc  = 1e-09
max_iter = 20
cal_stenosis = True
zero_cap = False

cal_params = {
        "tolerance_gradient": tol_grad,
        "tolerance_increment": tol_inc,
        "maximum_iterations": max_iter,
        "calibrate_stenosis_coefficient": cal_stenosis,
        "set_capacitance_to_zero": zero_cap
}

######################
# WRITE TO JSON FILE #
######################

obj = {"boundary_conditions": b_list, "junction": j_list, "simulation_parameters": sim_params, "vessels": v_list}
# obj = {"boundary_conditions": b_list, "junction": j_list, "simulation_parameters": sim_params, "vessels": v_list, "calibration_parameters": cal_params, }

# write to json file
with open('test_data.json', 'w') as json_file:
    # where data is your valid python dictionary
    json.dump(obj, json_file, indent=4)
    
    