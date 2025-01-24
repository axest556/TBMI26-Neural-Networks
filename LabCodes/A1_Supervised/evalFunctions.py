import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    correct = np.sum(LPred == LTrue)
    total = LPred.shape[0]
    acc = correct / total
    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------

    classes = np.unique(LTrue) # All possible labels
    nClasses = classes.shape[0] # nr of unique classes/labels

    # Initialize the confusion matrix
    cM = np.zeros((nClasses, nClasses), dtype=int)

    # Populate the confusion matrix
    for pred, true in zip(LPred, LTrue):
        # Find the index of the predicted and true labels
        pred_idx = np.where(classes == pred)[0][0] # Find row index in CM to increment cell
        true_idx = np.where(classes == true)[0][0] # Find column index
        
        # Increment the corresponding cell
        cM[pred_idx, true_idx] += 1
        
    # ============================================

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    # Nr of cases where the predicted label matches the true label
    true = np.trace(cM)

    # Sum of all elements (total predictions)
    all = np.sum(cM)

    # Calculate accuracy
    acc = true / all
    
    # ============================================

    return acc
