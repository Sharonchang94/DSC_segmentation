import numpy as np

def cal_dice(preds, labs):
    eps = 1e-10
    tp = ((preds == labs) & (preds == 1)).sum()  # Calculate true positives
    fp = ((preds != labs) & (preds == 1)).sum()  # Calculate false positives
    fn = ((preds != labs) & (preds == 0)).sum()  # Calculate false negatives
    dice = (tp * 2 + eps) / (tp * 2 + fn + fp + eps)  # Calculate Dice coefficient

    return dice  # Return Dice coefficient

# Load predicted mask
preds_path = "path_to_preds.npy"
preds = np.load(preds_path)

# Load ground truth mask
labs_path = "path_to_labs.npy"
labs = np.load(labs_path)

# Set non-zero values to 1
preds[preds != 0] = 1
labs[labs != 0] = 1

# Calculate Dice coefficient
dice = cal_dice(preds, labs)

# Print the result
print("Dice coefficient:", dice)
