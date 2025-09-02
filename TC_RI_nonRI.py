# Concatenate the test set data by each TC
import numpy as np

# all_trues and all_preds are obtained from Train_test.py
# sid_test is a list that stores all the sids, obtained from Data_processing.py

# Initialize dictionaries to store the predictions and true values for each typhoon
trues_by_sid = {}
preds_by_sid = {}

for sid, true_vals, pred_vals in zip(sid_test, all_trues_inversed, all_preds_inversed):
    if sid not in trues_by_sid:
        trues_by_sid[sid] = []
        preds_by_sid[sid] = []
    
    trues_by_sid[sid].append(true_vals)
    preds_by_sid[sid].append(pred_vals)

# Convert the values in the dictionary to np.array
trues_by_sid = {sid: np.array(vals) for sid, vals in trues_by_sid.items()} #56
preds_by_sid = {sid: np.array(vals) for sid, vals in preds_by_sid.items()}

# Print the results if needed for inspection
for sid in trues_by_sid:
    print(f'TC {sid} - True Values Shape: {trues_by_sid[sid].shape}, Predicted Values Shape: {preds_by_sid[sid].shape}')


#%%
import numpy as np

ri_trues_sids = []
ri_preds_sids = []

# Additional counters + lists to record TCs detected by "Method 1"
pred_count_method1 = 0
pred_count_method2 = 0
pred_sids_method1 = []  # Record which TCs are detected as RI by Method 1

for sid in trues_by_sid.keys():
    true_vals = trues_by_sid[sid]
    pred_vals = preds_by_sid[sid]
    
    # Method 1: Use the future 24-hour intensity change data
    true_ri_flag_method1 = np.any(true_vals[:, -1] >= 14)
    pred_ri_flag_method1 = np.any(pred_vals[:, -1] >= 14)
    
    # Method 2: Use past 24-hour wind speed change to determine RI
    true_ri_flag_method2 = False
    pred_ri_flag_method2 = False
    
    for i in range(4, len(true_vals)):
        # Check if the wind speed change at any point within 24 hours exceeds the threshold
        if (true_vals[i, 2] - true_vals[i-4, 2] >= 14 or  # Difference between the 5th and 1st point
            true_vals[i-1, 2] - true_vals[i-4, 2] >= 14 or  # Difference between the 4th and 1st point
            true_vals[i-2, 2] - true_vals[i-4, 2] >= 14 or  # Difference between the 3rd and 1st point
            true_vals[i-3, 2] - true_vals[i-4, 2] >= 14):   # Difference between the 2nd and 1st point
            true_ri_flag_method2 = True
            break
    
    for i in range(4, len(pred_vals)):
        if (pred_vals[i, 2] - pred_vals[i-4, 2] >= 14 or  # Difference between the 5th and 1st point
            pred_vals[i-1, 2] - pred_vals[i-4, 2] >= 14 or  # Difference between the 4th and 1st point
            pred_vals[i-2, 2] - pred_vals[i-4, 2] >= 14 or  # Difference between the 3rd and 1st point
            pred_vals[i-3, 2] - pred_vals[i-4, 2] >= 14):   # Difference between the 2nd and 1st point
            pred_ri_flag_method2 = True
            break

    # For true data, require AND condition
    if true_ri_flag_method1 and true_ri_flag_method2:
        ri_trues_sids.append(sid)
    
    # Count how many TCs are detected by model Method 1 and Method 2
    if pred_ri_flag_method1:
        pred_count_method1 += 1
        pred_sids_method1.append(sid)  # Record TCs identified as RI by Method 1
    if pred_ri_flag_method2:
        pred_count_method2 += 1

    # Final predicted RI: Method 1 AND Method 2
    if pred_ri_flag_method1 and pred_ri_flag_method2:
        ri_preds_sids.append(sid)

print("True data RI SIDs count:", len(ri_trues_sids))
print("Predicted data RI SIDs (AND condition) count:", len(ri_preds_sids))
print("Predicted RI count by Method1 (Future 24h Intensity Change):", pred_count_method1)
print("Predicted RI count by Method2 (Past 24h Wind Speed Change):", pred_count_method2)

# Find TCs that truly underwent RI but were not detected by model Method 1
lost_method1_sids = set(ri_trues_sids) - set(pred_sids_method1)
print("\nThese TCs are real RI but not detected by model method1:\n", lost_method1_sids)

# Print maximum predicted future 24-hour wind speed change (pred_vals[:, -1]) for these missed TCs
for sid in lost_method1_sids:
    pred_vals = preds_by_sid[sid]
    max_pred_change = np.max(pred_vals[:, -1])
    print(f"TC {sid}: max predicted 24h intensity change = {max_pred_change:.3f}")


#%%
# Add after the previous code

# 1) Collect all SIDs into a set for statistics
all_sids_set = set(trues_by_sid.keys())  # Alternatively, use set(sid_test) if sid_test covers all TCs

# Convert RI SID lists into sets for set operations
ri_trues_set = set(ri_trues_sids)
ri_preds_set = set(ri_preds_sids)

# 2) Count TP, FN, FP, TN
#   TP: True RI and predicted RI
tp = len(ri_trues_set & ri_preds_set)  
#   FN: True RI but not predicted as RI
fn = len(ri_trues_set - ri_preds_set)  
#   FP: Not true RI, but predicted as RI
fp = len(ri_preds_set - ri_trues_set)  
#   TN: Neither true RI nor predicted RI
tn = len(all_sids_set - (ri_trues_set | ri_preds_set))

print(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")

# 3) Calculate Heidke Skill Score (HSS)
def heidke_skill_score(tp, tn, fp, fn):
    """
    Calculate Heidke Skill Score (HSS)
    TP: True Positive
    TN: True Negative
    FP: False Positive
    FN: False Negative
    """
    N = tp + tn + fp + fn
    if N == 0:
        return float('nan')  # avoid division by zero

    # Accuracy
    ACC = (tp + tn) / N

    # Standard Forecast (SF)
    # Reference HSS formula:
    # SF = (TP+FN)/N * (TP+FP)/N + (TN+FN)/N * (TN+FP)/N
    SF = ((tp + fn) / N) * ((tp + fp) / N) + ((tn + fn) / N) * ((tn + fp) / N)

    # HSS
    if (1 - SF) == 0:
        return float('nan')  # avoid division by zero
    HSS = (ACC - SF) / (1 - SF)
    return HSS

hss = heidke_skill_score(tp, tn, fp, fn)
print(f"Heidke Skill Score (HSS): {hss:.3f}")

#%%
# View all test SIDs
# print("All test SIDs:")
# print(sorted(set(sid_test)))  # Deduplicate and sort

# View true RI SIDs
# print("True RI SIDs:")
# print(sorted(set(ri_trues_sids)))

# View predicted RI SIDs
print("Predicted RI SIDs:")

# ---- New: three additional metrics ----

def calc_metrics(tp, fn, fp, tn):
    # 1. Probability of Detection (POD) = TP / (TP + FN)
    pod = tp / (tp + fn) if (tp + fn) > 0 else float('nan')

    # 2. False Alarm Rate (FARate) = FP / (FP + TN)
    farate = fp / (fp + tn) if (fp + tn) > 0 else float('nan')

    # 3. False Alarm Ratio (FARatio) = FP / (TP + FP)
    faratio = fp / (tp + fp) if (tp + fp) > 0 else float('nan')

    return pod, farate, faratio

pod, farate, faratio = calc_metrics(tp, fn, fp, tn)
print(f"Probability of Detection (POD): {pod:.3f}")
print(f"False Alarm Rate (FARate)     : {farate:.3f}")
print(f"False Alarm Ratio (FARatio)   : {faratio:.3f}")

print(sorted(set(ri_preds_sids)))


#%%
import xarray as xr

# 1) Read the NetCDF file and extract typhoon IDs and names
path = '/Users/lzh/Desktop/'
ds = xr.open_dataset(path + 'IBTrACS.WP.v04r01.nc')

typhoon_ids = [id.decode('utf-8') for id in ds['sid'].values]
typhoon_names = [name.decode('utf-8') for name in ds['name'].values]

# 2) Define a function to return typhoon names (with year) from a given SID list
def get_typhoon_names_from_ids(target_ids, ty_ids, ty_names):
    """
    For each SID in target_ids, find its index in ty_ids,
    then get the corresponding name from ty_names and append year information.
    Returns a list where each element is in the format 'HAISHEN (2020)'.
    """
    result_list = []
    for sid in target_ids:
        if sid in ty_ids:
            index = ty_ids.index(sid)
            ty_name = ty_names[index]
            year = sid[0:4]  # The first 4 characters of SID usually represent the year
            formatted_name = f"{ty_name} ({year})"
            result_list.append(formatted_name)
        else:
            print(f"Typhoon name not found for ID {sid}.")
    return result_list

# 3) Prepare your existing SID lists
# all_ids = sid_test
# ri_trues_sids
# ri_preds_sids
# Here, we use variable names as examples. Replace them with your actual variables.

all_ids_list = list(set(sid_test))    # All test set typhoon IDs
ri_trues_list = list(ri_trues_sids)   # True RI typhoon IDs
ri_preds_list = list(ri_preds_sids)   # Predicted RI typhoon IDs

# 4) Retrieve the corresponding typhoon name lists
all_test_typhoons = get_typhoon_names_from_ids(all_ids_list, typhoon_ids, typhoon_names)
ri_true_typhoons = get_typhoon_names_from_ids(ri_trues_list, typhoon_ids, typhoon_names)
ri_pred_typhoons = get_typhoon_names_from_ids(ri_preds_list, typhoon_ids, typhoon_names)

# 5) Print or further process the three variables
print("All test typhoons:")
print(all_test_typhoons)

print("\nTrue RI typhoons:")
print(ri_true_typhoons)

print("\nPred RI typhoons:")
print(ri_pred_typhoons)



