#%%
# Display all 5 target variables in one figure
# DeepLIFT for RI, Intensity, MSLP, Latitude, and Longitude
import torch
import numpy as np
import joblib
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import DeepLift
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import pickle  # Ensure the pickle library is imported

# Load data and model
path = '/Users/lzh/Desktop/'
with open(path + 'test_dataset.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

# Load the scaler for the test set
scaler_y_test = joblib.load(path + 'y_test_scaler.pkl')

# Convert to PyTorch dataset
batch_size = 128
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the pre-trained model
model.load_state_dict(torch.load(path + 'TF_checkpoint_F2.pt'))
model.eval()

# Use DeepLift method from Captum
deeplift = DeepLift(model)

# Target variables and their corresponding titles
targets = [(0, 4), (0, 2), (0, 3), (0, 0), (0, 1)]
titles = [
    '(a) Feature Importance for Future 24-h TC Intensity Change using DeepLIFT',
    '(b) Feature Importance for Future 6-h Intensity using DeepLIFT',
    '(c) Feature Importance for Future 6-h MSLP using DeepLIFT',
    '(d) Feature Importance for Future 6-h Latitude using DeepLIFT',
    '(e) Feature Importance for Future 6-h Longitude using DeepLIFT'
]

# Define feature names
feature_names = ['Lat', 'Lon', 'U_steering', 'V_steering','Mov_angle', 'Trans_speed', 'Intensity', 'dUmax24', 'MSLP', 
                 'OWZ_850', 'OWZ_500', 'RH_925', 'RH_700', 'VWS_200-850', 'SH_925']

# Define color mapping, categorized by types
colors = {
    'Basic factors': 'skyblue',
    'Gradient factors': 'lightcoral',
    'Structural factors': 'lightgreen',
    'Scale environmental factors': 'gold'
}

# Define the category for each variable
categories = {
    'Lat': 'Basic factors',
    'Lon': 'Basic factors',
    'Intensity': 'Basic factors',
    'MSLP': 'Basic factors',
    'U_steering':'Gradient factors',
    'V_steering':'Gradient factors',    
    'Trans_speed': 'Gradient factors',
    'Mov_angle': 'Gradient factors',
    'dUmax24': 'Gradient factors',
    'OWZ_850': 'Structural factors',
    'OWZ_500': 'Structural factors',
    'VWS_200-850': 'Scale environmental factors',
    'RH_925': 'Scale environmental factors',
    'RH_700': 'Scale environmental factors',
    'SH_925': 'Scale environmental factors'
}

# Create subplots, 5 rows and 1 column, adjust font and legend
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 18))

# Set font size
plt.rcParams.update({'font.size': 8})

for idx, (target, title) in enumerate(zip(targets, titles)):
    # Compute attribution values
    attr = deeplift.attribute(X_test_tensor, target=target)

    # Compute mean attribution values
    mean_attr = attr.mean(dim=(0, 1)).detach().numpy()

    # Convert feature importance values to absolute values
    abs_mean_attr = np.abs(mean_attr)
    
    # abs_mean_attr = attr.abs().mean(dim=(0, 1)).detach().numpy()

    # Sort features by absolute importance
    sorted_indices = np.argsort(abs_mean_attr)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importances = abs_mean_attr[sorted_indices]

    # Create a color list, ordered by categories
    color_list = [colors[categories[feature]] for feature in sorted_features]

    # Plot the bar chart
    bars = axes[idx].barh(sorted_features[::-1], sorted_importances[::-1], color=color_list[::-1])  # Reverse order
    axes[idx].set_xlabel('Mean Absolute Contribution Score', fontsize=12)
    axes[idx].set_title(title, fontsize=16, loc='left')

    # Adjust the font size for the y-axis labels
    axes[idx].tick_params(axis='y', labelsize=10)

    # Label values in the bar plot
    for bar, value in zip(bars, sorted_importances[::-1]):
        axes[idx].text(value, bar.get_y() + bar.get_height() / 2, f'{value:.4f}', 
                      va='center', ha='left', color='black', fontsize=8)

    # Add legend and adjust its font size
    legend_patches = [mpatches.Patch(color=color, label=category) for category, color in colors.items()]
    axes[idx].legend(handles=legend_patches, title='Category', loc='lower right', fontsize=8, title_fontsize=9)

# Adjust spacing between subplots
plt.tight_layout(pad=2.0)
plt.savefig(path + 'deeplift.pdf', format='pdf', bbox_inches='tight')
plt.savefig(path + 'deeplift.png', dpi=300, bbox_inches='tight')

plt.show()


#%%
# Display all 5 target variables in one figure
# DeepLiftShap for RI, Intensity, MSLP, Latitude, and Longitude with multiple baselines
import torch
import numpy as np
import joblib
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import DeepLiftShap
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import pickle

# Load data and model
path = '/Users/lzh/Desktop/'
with open(path + 'test_dataset.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

# Load the scaler for the test set
scaler_y_test = joblib.load(path + 'y_test_scaler.pkl')

# Convert to PyTorch dataset
batch_size = 8  # Reduce batch_size to lower memory usage
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the pre-trained model
model.load_state_dict(torch.load(path+'TF_checkpoint_F2.pt'))
model.eval()

# Use DeepLiftShap method from Captum
deeplift_shap = DeepLiftShap(model)

# Baseline 1: All zeros
baseline_zeros = torch.zeros_like(X_test_tensor)

# Baseline 2: Mean of input features
baseline_means = torch.mean(X_test_tensor, dim=0, keepdim=True)

# Baseline 3: Randomly sampled input
baseline_random = X_test_tensor[torch.randint(0, len(X_test_tensor), (1,))]

# Combine the three baselines into one batch
baselines = torch.cat([baseline_zeros, baseline_means, baseline_random], dim=0)

# Target variables and their titles (RI, Intensity, MSLP, Latitude, Longitude)
targets = [(0, 4), (0, 2), (0, 3), (0, 0), (0, 1)]
titles = [
    '(a) Feature Importance for Future 24-h TC Intensity Change using DeepLiftShap',
    '(b) Feature Importance for Future 6-h Intensity using DeepLiftShap',
    '(c) Feature Importance for Future 6-h MSLP using DeepLiftShap',
    '(d) Feature Importance for Future 6-h Latitude using DeepLiftShap',
    '(e) Feature Importance for Future 6-h Longitude using DeepLiftShap'
]

# Define feature names
feature_names = ['Lat', 'Lon', 'U_steering', 'V_steering','Mov_angle', 'Trans_speed', 'Intensity', 'dUmax24', 'MSLP', 
                 'OWZ_850', 'OWZ_500', 'RH_925', 'RH_700', 'VWS_200-850', 'SH_925']

# Define color mapping, categorized by types
colors = {
    'Basic factors': 'skyblue',
    'Gradient factors': 'lightcoral',
    'Structural factors': 'lightgreen',
    'Scale environmental factors': 'gold'
}

# Define the category for each variable
categories = {
    'Lat': 'Basic factors',
    'Lon': 'Basic factors',
    'Intensity': 'Basic factors',
    'MSLP': 'Basic factors',
    'U_steering':'Gradient factors',
    'V_steering':'Gradient factors',    
    'Trans_speed': 'Gradient factors',
    'Mov_angle': 'Gradient factors',
    'dUmax24': 'Gradient factors',
    'OWZ_850': 'Structural factors',
    'OWZ_500': 'Structural factors',
    'VWS_200-850': 'Scale environmental factors',
    'RH_925': 'Scale environmental factors',
    'RH_700': 'Scale environmental factors',
    'SH_925': 'Scale environmental factors'
}

# Create subplots, 5 rows and 1 column, adjust font and legend
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 18))

# Set font size
plt.rcParams.update({'font.size': 8})

for idx, (target, title) in enumerate(zip(targets, titles)):
    # Initialize an array to store batch results
    all_attr = []

    # Compute attribution values in batches
    for batch in test_loader:
        X_batch, _ = batch
        # Pass multiple baselines to compute attribution
        attr_batch = deeplift_shap.attribute(X_batch, baselines=baselines, target=target)
        all_attr.append(attr_batch)

    # Combine results from all batches
    all_attr = torch.cat(all_attr, dim=0)

    # Compute mean attribution values
    mean_attr = all_attr.mean(dim=(0, 1)).detach().numpy()

    # Convert feature importance values to absolute values
    abs_mean_attr = np.abs(mean_attr)

    # Sort features by absolute importance
    sorted_indices = np.argsort(abs_mean_attr)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importances = abs_mean_attr[sorted_indices]

    # Create a color list, ordered by categories
    color_list = [colors[categories[feature]] for feature in sorted_features]

    # Plot the bar chart
    bars = axes[idx].barh(sorted_features[::-1], sorted_importances[::-1], color=color_list[::-1])  # Reverse order
    axes[idx].set_xlabel('Mean Absolute Shapley Value', fontsize=12)
    axes[idx].set_title(title, fontsize=16, loc='left')

    # Adjust the font size for the y-axis labels
    axes[idx].tick_params(axis='y', labelsize=10)

    # Label values in the bar plot
    for bar, value in zip(bars, sorted_importances[::-1]):
        axes[idx].text(value, bar.get_y() + bar.get_height() / 2, f'{value:.4f}', 
                      va='center', ha='left', color='black', fontsize=8)

    # Add legend and adjust its font size
    legend_patches = [mpatches.Patch(color=color, label=category) for category, color in colors.items()]
    axes[idx].legend(handles=legend_patches, title='Category', loc='lower right', fontsize=8, title_fontsize=9)

# Adjust spacing between subplots
plt.tight_layout(pad=2.0)
plt.savefig(path + 'deepliftshap.pdf', format='pdf', bbox_inches='tight')
plt.savefig(path + 'deepliftshap.png', dpi=300, bbox_inches='tight')
plt.show()