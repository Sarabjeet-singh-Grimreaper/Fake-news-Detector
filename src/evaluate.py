import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, model_name):
    """
    Generates and prints a detailed classification report, 
    and visually plots and saves a confusion matrix heatmap.
    """
    print(f"\n[Eval] --- Detailed Evaluation Metrics for {model_name} ---")
    # Prints Precision, Recall, and F1-score for both 0 (Fake) and 1 (Real)
    print(classification_report(y_true, y_pred))
    
    # Calculate confusion matrix array
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot using Seaborn
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Fake (0)', 'Real (1)'], 
        yticklabels=['Fake (0)', 'Real (1)']
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Create a reports folder directory structurally if it doesn't exist
    os.makedirs("reports/plots", exist_ok=True)
    
    # Save chart file cleanly to disk
    safe_filename = model_name.lower().replace(" ", "_")
    plot_path = f"reports/plots/{safe_filename}_confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close() # Close plot to free up machine memory
    
    print(f"[Eval] Confusion matrix plot saved successfully to: {plot_path}")