import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class DimensionalityReductionComparison:
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size
        self.data = {}
        self.scaler = StandardScaler()
        
    def _preprocess_image(self, img):
        """Consistent image preprocessing"""
        # Resize image
        img_resized = cv2.resize(img, self.target_size)
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        # Flatten
        return img_normalized.flatten()
    
    def load_dataset(self, base_dir):
        """Load thermal image dataset"""
        print(f"Loading data from {base_dir}...")
        filters = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        
        for filter_name in filters:
            filter_path = os.path.join(base_dir, filter_name)
            print(f"Processing filter: {filter_name}")
            
            X, y = [], []
            stages = [s for s in os.listdir(filter_path) if os.path.isdir(os.path.join(filter_path, s))]
            stages.sort()
            
            for stage_name in stages:
                stage_path = os.path.join(filter_path, stage_name)
                stage_images = 0
                
                for img_name in os.listdir(stage_path):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        continue
                    
                    img_path = os.path.join(stage_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    flat = self._preprocess_image(img)
                    X.append(flat)
                    y.append(stage_name)
                    stage_images += 1
                
                print(f"  {stage_name}: {stage_images} images")
            
            self.data[filter_name] = {
                'X': np.array(X),
                'y': np.array(y),
                'stages': stages
            }
        
        return self.data
    
    def compare_pca_umap(self, filter_name):
        """Compare PCA 2D projection with UMAP 2D embedding"""
        if filter_name not in self.data:
            print(f"Filter {filter_name} not found in loaded data")
            return
        
        X = self.data[filter_name]['X']
        y = self.data[filter_name]['y']
        
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Apply UMAP
        umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        X_umap = umap_model.fit_transform(X_scaled)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'PCA vs UMAP Comparison: {filter_name}', fontsize=16, fontweight='bold')
        
        unique_stages = np.unique(y)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_stages)))
        
        # PCA 2D Projection
        for i, stage in enumerate(unique_stages):
            mask = y == stage
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c=[colors[i]], label=stage, alpha=0.7, s=40)
        
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f} variance)')
        axes[0].set_title('PCA: 2D Projection')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # UMAP 2D Embedding
        for i, stage in enumerate(unique_stages):
            mask = y == stage
            axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1], 
                           c=[colors[i]], label=stage, alpha=0.7, s=40)
        
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        axes[1].set_title('UMAP: 2D Embedding')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Calculate and print metrics
        try:
            pca_silhouette = silhouette_score(X_pca, y)
            umap_silhouette = silhouette_score(X_umap, y)
            
            print(f"\nSeparability Metrics for {filter_name}:")
            print(f"PCA Silhouette Score: {pca_silhouette:.4f}")
            print(f"UMAP Silhouette Score: {umap_silhouette:.4f}")
            print(f"PCA Explained Variance: {pca.explained_variance_ratio_.sum():.4f}")
            print(f"Better separability: {'UMAP' if umap_silhouette > pca_silhouette else 'PCA'}")
        except:
            print("Could not calculate silhouette scores")
        
        return X_pca, X_umap, pca, umap_model
    
    def analyze_all_filters(self, base_dir):
        """Analyze all filters and create comparisons"""
        self.load_dataset(base_dir)
        
        for filter_name in self.data.keys():
            print(f"\n{'='*50}")
            print(f"Analyzing Filter: {filter_name}")
            print(f"{'='*50}")
            
            X = self.data[filter_name]['X']
            y = self.data[filter_name]['y']
            
            print(f"Data shape: {X.shape}")
            print(f"Number of stages: {len(np.unique(y))}")
            print(f"Stages: {list(np.unique(y))}")
            
            # Create comparison plot
            self.compare_pca_umap(filter_name)
            plt.show()
    
    def create_combined_comparison(self, base_dir):
        """Create a combined plot showing all filters"""
        self.load_dataset(base_dir)
        
        n_filters = len(self.data)
        fig, axes = plt.subplots(n_filters, 2, figsize=(12, 4*n_filters))
        if n_filters == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('PCA vs UMAP: All Filters Comparison', fontsize=16, fontweight='bold')
        
        for idx, filter_name in enumerate(self.data.keys()):
            X = self.data[filter_name]['X']
            y = self.data[filter_name]['y']
            
            # Standardize data
            X_scaled = self.scaler.fit_transform(X)
            
            # Apply PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Apply UMAP
            umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
            X_umap = umap_model.fit_transform(X_scaled)
            
            unique_stages = np.unique(y)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_stages)))
            
            # PCA Plot
            for i, stage in enumerate(unique_stages):
                mask = y == stage
                axes[idx, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                                   c=[colors[i]], label=stage, alpha=0.7, s=30)
            
            axes[idx, 0].set_title(f'{filter_name} - PCA')
            axes[idx, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            axes[idx, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            axes[idx, 0].legend(loc='best', fontsize=8)
            axes[idx, 0].grid(True, alpha=0.3)
            
            # UMAP Plot
            for i, stage in enumerate(unique_stages):
                mask = y == stage
                axes[idx, 1].scatter(X_umap[mask, 0], X_umap[mask, 1], 
                                   c=[colors[i]], label=stage, alpha=0.7, s=30)
            
            axes[idx, 1].set_title(f'{filter_name} - UMAP')
            axes[idx, 1].set_xlabel('UMAP 1')
            axes[idx, 1].set_ylabel('UMAP 2')
            axes[idx, 1].legend(loc='best', fontsize=8)
            axes[idx, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Usage example
def main():
    # Update this path to match your directory structure
    base_dir = r"C:\Users\Asus\Desktop\DATASETS_FOR_FILTERS\Filtered_Outputs"
    
    # Initialize the comparison class
    comparator = DimensionalityReductionComparison(target_size=(64, 64))
    
    # Option 1: Analyze each filter separately
    print("Analyzing each filter separately...")
    comparator.analyze_all_filters(base_dir)
    
    # Option 2: Create combined comparison plot
    print("\nCreating combined comparison plot...")
    comparator.create_combined_comparison(base_dir)
    
    # Option 3: Analyze specific filter only
    # comparator.load_dataset(base_dir)
    # comparator.compare_pca_umap('specific_filter_name')

if __name__ == "__main__":
    main()