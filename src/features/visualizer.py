import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


def plot_low_level_feature_dist(df, feature_name):
    plot_data = []
    categories = []

    for idx, row in df.iterrows():
        feature_array = row[feature_name]
        real_or_fake = row['real_or_fake']
        plot_data.extend(feature_array)
        categories.extend([real_or_fake] * len(feature_array))

    plot_df = pd.DataFrame({
        'value': plot_data,
        'category': categories
    })

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='category', y='value', data=plot_df)
    plt.title(f'Boxplot of {feature_name} by Category')
    plt.xlabel('Category')
    plt.ylabel('Value')

    plt.subplot(1, 2, 2)
    sns.violinplot(x='category', y='value', data=plot_df)
    plt.title(f'Violin Plot of {feature_name} by Category')
    plt.xlabel('Category')
    plt.ylabel('Value')

    plt.tight_layout()
    plt.show()


def plot_high_level_feature_dist(df, feature_list, target_column='real_or_fake'):
    for feature in feature_list:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(x=target_column, y=feature, data=df)
        plt.title(f'Boxplot of {feature} by Category')
        plt.xlabel('Category')
        plt.ylabel(feature)

        plt.subplot(1, 2, 2)
        sns.violinplot(x=target_column, y=feature, data=df)
        plt.title(f'Violin Plot of {feature} by Category')
        plt.xlabel('Category')
        plt.ylabel(feature)

        plt.tight_layout()
        plt.show()
        
        
def perform_pca_and_plot(df, selected_features, target_column='real_or_fake'):
    # Standardize the features
    features = df[selected_features]
    
    # Handle missing values by imputation
    imputer = SimpleImputer(strategy='mean')
    imputed_features = imputer.fit_transform(features)
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(imputed_features)

    # Apply PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    
    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df[target_column] = df[target_column].values

    # Plot the PCA results
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue=target_column, data=pca_df, palette='Set2')
    plt.title('PCA of High-Level Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title=target_column)
    plt.show()