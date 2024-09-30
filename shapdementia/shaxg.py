!pip install shap

import shap
from import matplotlib.pyplot as plt

# SHAP values computation for Decision Tree
explainer = shap.TreeExplainer(xg_classifier)
shap_values = explainer.shap_values(X)

# Loop through each class and create SHAP summary plots for each
class_names = encoder.classes_
for i, class_name in enumerate(class_names):
    shap.summary_plot(shap_values[:, :, i], X, plot_type="bar", feature_names=X.columns)


# Mean absolute SHAP values for each feature and class
mean_abs_shap_values = np.abs(shap_values).mean(axis=0)  # Shape will be (num_features, num_classes)

# Create DataFrame for easier plotting
shap_df = pd.DataFrame(mean_abs_shap_values, columns=['Converted', 'Demented', 'Nondemented'], index=X.columns)

# Calculate total importance for sorting
shap_df['Total_Importance'] = shap_df.sum(axis=1)

# Sort the DataFrame by Total_Importance in ascending order
shap_df_sorted = shap_df.sort_values(by='Total_Importance', ascending=True)

# Create a DataFrame to hold feature importance values
feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Mean Absolute SHAP Value': mean_abs_shap_values[:,0]
    }).sort_values(by='Mean Absolute SHAP Value', ascending=False)

# Create a DataFrame to hold feature importance values
feature_importance_df1 = pd.DataFrame({
        'Feature': X.columns,
        'Mean Absolute SHAP Value': mean_abs_shap_values[:,1]
    }).sort_values(by='Mean Absolute SHAP Value', ascending=False)

# Create a DataFrame to hold feature importance values
feature_importance_df2 = pd.DataFrame({
        'Feature': X.columns,
        'Mean Absolute SHAP Value': mean_abs_shap_values[:,2]
    }).sort_values(by='Mean Absolute SHAP Value', ascending=False)

 # Print the feature importance values
print("\nSHAP Feature Importance Values0:")
print(feature_importance_df)

print("\nSHAP Feature Importance Values1:")
print(feature_importance_df1)

print("\nSHAP Feature Importance Values2:")
print(feature_importance_df2)

# Plotting the sorted stacked bar chart with enhancements
fig, ax = plt.subplots(figsize=(12, 7))

# Plot stacked bar chart
shap_df_sorted.drop(columns='Total_Importance').plot(
    kind='barh', stacked=True, ax=ax,
    color=['mediumorchid', 'crimson', 'cornflowerblue'],
    width=0.8, edgecolor='black', linewidth=0.5
)

# Adding grid lines for both x and y axes
ax.grid(axis='x', linestyle='--', alpha=0.6)
ax.grid(axis='y', linestyle=':', alpha=0.4)

# Adding data labels to each bar for better readability
for bars in ax.containers:
    ax.bar_label(bars, fmt='%.2f', label_type='center', fontsize=9, color='black')

# Setting labels and title with enhanced styling
plt.title('Feature-wise Mean Absolute SHAP Values for All Classes using XGBoost', fontsize=16, color='darkblue')
plt.xlabel('Mean Absolute SHAP Value', fontsize=12, color='darkgreen')
plt.ylabel('Features', fontsize=12, color='darkgreen')

# Customizing tick marks for better readability
plt.xticks(fontsize=10, color='black')
plt.yticks(fontsize=10, color='black')

# Adding legend with custom styling
plt.legend(title='Class', labels=['Converted', 'Demented', 'Nondemented'], fontsize=10, title_fontsize=11, loc='lower right', edgecolor='black')

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
