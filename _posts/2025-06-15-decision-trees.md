---
layout: post
title: "Decision Trees: The Art of 20 Questions with Data"
date: 2025-06-15
categories: [machine-learning, classification, regression]
excerpt: "Discover how decision trees make predictions by asking a series of simple questions, and learn why they're one of the most intuitive machine learning algorithms"
---

# Decision Trees: The Art of 20 Questions with Data 🌳

## The Big Idea: Divide and Conquer

Remember playing "20 Questions" as a kid? You'd think of something, and your friend would ask yes/no questions to figure out what it was:
- "Is it alive?" → No
- "Is it bigger than a breadbox?" → Yes  
- "Is it electronic?" → Yes
- "Is it a computer?" → Yes!

That's exactly how decision trees work! They learn to ask the right questions about your data to make predictions. Each question splits the data into smaller groups until we can make a confident decision.

## A Visual Introduction 🎨

Let's say we're predicting if someone will play golf based on the weather:

```
                    [Play Golf?]
                         |
                   Outlook = ?
                /        |        \
             Sunny    Overcast    Rainy
              |          |          |
         Humidity?    PLAY ✓    Windy?
         /      \               /     \
      High      Low          True    False
        |        |            |        |
      DON'T ✗  PLAY ✓      DON'T ✗  PLAY ✓
```

Each internal node asks a question, each branch represents an answer, and each leaf gives us a prediction!

## How Trees Make Decisions: The Algorithm 🧮

The core algorithm is beautifully simple:

1. **Start with all your data at the root**
2. **Find the best question to split the data**
3. **Create branches based on the answer**
4. **Repeat for each branch until you reach a stopping condition**

But wait... what makes a question "best"? That's where the math comes in!

## The Mathematics: Measuring Disorder 📊

### Entropy: The Chaos Metric

Entropy measures how "mixed up" or "impure" a group is:

```
Entropy(S) = -Σ(p_i × log₂(p_i))
```

Where:
- `S` = the dataset
- `p_i` = proportion of samples belonging to class i

**Intuition:** 
- All samples same class → Entropy = 0 (perfectly pure)
- 50/50 split → Entropy = 1 (maximum chaos)

Example:
```python
import numpy as np

def entropy(y):
    """Calculate entropy of a label array"""
    # Get unique classes and their counts
    _, counts = np.unique(y, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts / len(y)
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    return entropy

# Examples
pure = [1, 1, 1, 1]  # All same class
print(f"Pure entropy: {entropy(pure):.3f}")  # 0.000

mixed = [1, 0, 1, 0]  # 50/50 split
print(f"Mixed entropy: {entropy(mixed):.3f}")  # 1.000

mostly_one = [1, 1, 1, 0]  # 75/25 split
print(f"Mostly one class: {entropy(mostly_one):.3f}")  # 0.811
```

### Information Gain: Finding the Best Questions

Information Gain tells us how much a question reduces entropy:

```
IG(S, A) = Entropy(S) - Σ((|S_v|/|S|) × Entropy(S_v))
```

Where:
- `A` = the attribute/question we're testing
- `S_v` = subset of S where attribute A has value v
- `|S_v|/|S|` = proportion of samples with value v

The question with the highest information gain wins!

### Gini Impurity: The Alternative

Some prefer Gini impurity (used by default in scikit-learn):

```
Gini(S) = 1 - Σ(p_i²)
```

It's faster to compute (no logarithms) and often gives similar results.

## Implementation: Building a Decision Tree from Scratch 🛠️

Let's build a complete decision tree classifier:

```python
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class Node:
    """A node in the decision tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # Feature index for splitting
        self.threshold = threshold      # Threshold value for splitting
        self.left = left               # Left child
        self.right = right             # Right child
        self.value = value             # Prediction value (for leaf nodes)
    
    def is_leaf(self):
        return self.value is not None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_features = None
        self.n_classes = None
        
    def fit(self, X, y):
        """Build the decision tree"""
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y)
        
    def _build_tree(self, X, y, depth=0):
        """Recursively build the tree"""
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            # Create leaf node
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            # No good split found
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Create child splits
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(best_feature, best_threshold, left_child, right_child)
    
    def _best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(self.n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                # Calculate information gain
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, y, X_column, threshold):
        """Calculate information gain of a split"""
        # Parent entropy
        if self.criterion == 'entropy':
            parent_metric = self._entropy(y)
        else:  # gini
            parent_metric = self._gini(y)
        
        # Generate split
        left_indices = X_column <= threshold
        right_indices = X_column > threshold
        
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0
        
        # Weighted average of children
        n = len(y)
        n_left, n_right = np.sum(left_indices), np.sum(right_indices)
        
        if self.criterion == 'entropy':
            e_left = self._entropy(y[left_indices])
            e_right = self._entropy(y[right_indices])
            child_metric = (n_left/n) * e_left + (n_right/n) * e_right
        else:  # gini
            g_left = self._gini(y[left_indices])
            g_right = self._gini(y[right_indices])
            child_metric = (n_left/n) * g_left + (n_right/n) * g_right
        
        # Information gain
        ig = parent_metric - child_metric
        return ig
    
    def _entropy(self, y):
        """Calculate entropy"""
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
    def _gini(self, y):
        """Calculate Gini impurity"""
        proportions = np.bincount(y) / len(y)
        gini = 1 - np.sum([p**2 for p in proportions])
        return gini
    
    def _most_common_label(self, y):
        """Get most common class label"""
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def predict(self, X):
        """Make predictions for samples"""
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make a prediction"""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def print_tree(self, node=None, depth=0):
        """Print the tree structure"""
        if node is None:
            node = self.root
        
        if node.is_leaf():
            print(f"{'  ' * depth}Predict: Class {node.value}")
        else:
            print(f"{'  ' * depth}Feature_{node.feature} <= {node.threshold:.2f}?")
            print(f"{'  ' * depth}├─ True:")
            self.print_tree(node.left, depth + 1)
            print(f"{'  ' * depth}└─ False:")
            self.print_tree(node.right, depth + 1)

# Example usage
if __name__ == "__main__":
    # Create a simple dataset
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3,
                              n_redundant=0, n_classes=2, random_state=42)
    
    # Train our tree
    tree = DecisionTreeClassifier(max_depth=3, criterion='entropy')
    tree.fit(X, y)
    
    # Print tree structure
    print("Decision Tree Structure:")
    tree.print_tree()
    
    # Make predictions
    predictions = tree.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nTraining Accuracy: {accuracy:.3f}")
```

## Visualizing Decision Boundaries 🎨

Let's see how decision trees create those characteristic "boxy" decision boundaries:

```python
def visualize_decision_boundary(X, y, tree_model, title="Decision Tree Boundary"):
    """Visualize the decision boundary of a tree"""
    h = 0.02  # Step size in mesh
    
    # Create a mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = tree_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # Plot training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
                         edgecolor='black', s=50)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter)
    plt.show()

# Example with 2D data
X_2d, y_2d = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                 n_informative=2, n_clusters_per_class=1,
                                 class_sep=1.0, random_state=42)

tree_2d = DecisionTreeClassifier(max_depth=5)
tree_2d.fit(X_2d, y_2d)

visualize_decision_boundary(X_2d, y_2d, tree_2d)
```

## The Overfitting Problem: When Trees Grow Too Deep 🌲

Decision trees have a superpower and a weakness - they can perfectly memorize training data:

```python
def demonstrate_overfitting():
    """Show how tree depth affects overfitting"""
    from sklearn.model_selection import train_test_split
    
    # Generate data
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1,
                              flip_y=0.1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    depths = [1, 3, 5, 10, None]  # None = no limit
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for idx, depth in enumerate(depths):
        ax = axes[idx]
        
        # Train tree
        tree = DecisionTreeClassifier(max_depth=depth)
        tree.fit(X_train, y_train)
        
        # Calculate accuracies
        train_acc = tree.predict(X_train).mean() == y_train.mean()
        test_acc = tree.predict(X_test).mean() == y_test.mean()
        
        # Plot decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                  cmap=plt.cm.RdYlBu, edgecolor='black', s=50)
        
        depth_str = 'No limit' if depth is None else f'{depth}'
        ax.set_title(f'Max Depth: {depth_str}')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.show()

demonstrate_overfitting()
```

## Pruning: Trimming the Tree 🌿

To combat overfitting, we can prune our trees:

### 1. Pre-pruning (Early Stopping)
Stop growing the tree early:
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split a node
- `min_samples_leaf`: Minimum samples in a leaf
- `max_features`: Maximum features to consider

### 2. Post-pruning (Cost Complexity)
Grow full tree, then remove branches that don't improve validation performance:

```python
def cost_complexity_pruning_example():
    """Demonstrate cost complexity pruning"""
    from sklearn.tree import DecisionTreeClassifier as SklearnDT
    from sklearn.model_selection import train_test_split
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                              n_redundant=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Get pruning path
    tree = SklearnDT(random_state=42)
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    # Train trees with different alpha values
    trees = []
    for ccp_alpha in ccp_alphas:
        tree = SklearnDT(random_state=42, ccp_alpha=ccp_alpha)
        tree.fit(X_train, y_train)
        trees.append(tree)
    
    # Calculate accuracies
    train_scores = [tree.score(X_train, y_train) for tree in trees]
    test_scores = [tree.score(X_test, y_test) for tree in trees]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ccp_alphas, train_scores, marker='o', label='Train', drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label='Test', drawstyle="steps-post")
    ax.set_xlabel('Alpha (cost-complexity parameter)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Alpha for Post-Pruning')
    ax.legend()
    plt.show()

cost_complexity_pruning_example()
```

## Feature Importance: What Matters Most? 📊

Trees tell us which features are most important for predictions:

```python
def calculate_feature_importance(tree, X, feature_names=None):
    """Calculate and visualize feature importance"""
    from sklearn.tree import DecisionTreeClassifier as SklearnDT
    
    # Use sklearn for comparison
    sklearn_tree = SklearnDT(max_depth=5, random_state=42)
    sklearn_tree.fit(X, y)
    
    importances = sklearn_tree.feature_importances_
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance in Decision Tree')
    plt.tight_layout()
    plt.show()
    
    # Print ranking
    print("Feature Ranking:")
    for i in range(X.shape[1]):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
```

## Decision Trees for Regression 📈

Trees aren't just for classification! They can predict continuous values too:

```python
class DecisionTreeRegressor:
    """Simple regression tree implementation"""
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
    
    def _build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            # Return mean value for leaf
            return Node(value=np.mean(y))
        
        # Find best split (minimize MSE)
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return Node(value=np.mean(y))
        
        # Split data
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold
        
        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return Node(best_feature, best_threshold, left_child, right_child)
    
    def _best_split(self, X, y):
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                mse = self._mse_split(X[:, feature_idx], y, threshold)
                
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _mse_split(self, X_column, y, threshold):
        left_idx = X_column <= threshold
        right_idx = X_column > threshold
        
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return float('inf')
        
        # Calculate weighted MSE
        n = len(y)
        n_left = np.sum(left_idx)
        n_right = np.sum(right_idx)
        
        mse_left = np.mean((y[left_idx] - np.mean(y[left_idx]))**2)
        mse_right = np.mean((y[right_idx] - np.mean(y[right_idx]))**2)
        
        weighted_mse = (n_left/n) * mse_left + (n_right/n) * mse_right
        return weighted_mse
    
    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Demo regression tree
X_reg = np.linspace(0, 10, 100).reshape(-1, 1)
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.1, X_reg.shape[0])

reg_tree = DecisionTreeRegressor(max_depth=5)
reg_tree.fit(X_reg, y_reg)

plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, alpha=0.5, label='Data')
plt.plot(X_reg, reg_tree.predict(X_reg), color='red', linewidth=2, label='Tree Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()
```

## Random Forests: Better Together 🌲🌲🌲

Single trees are unstable - small changes in data can create very different trees. Random Forests fix this by combining many trees:

```python
def demonstrate_random_forest():
    """Show how Random Forests improve on single trees"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier as SklearnDT
    
    # Generate noisy data
    X, y = make_classification(n_samples=300, n_features=10, n_informative=5,
                              n_redundant=3, flip_y=0.1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Single tree
    single_tree = SklearnDT(max_depth=None, random_state=42)
    single_tree.fit(X_train, y_train)
    
    # Random Forest
    forest = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    forest.fit(X_train, y_train)
    
    print("Single Decision Tree:")
    print(f"  Train Accuracy: {single_tree.score(X_train, y_train):.3f}")
    print(f"  Test Accuracy: {single_tree.score(X_test, y_test):.3f}")
    
    print("\nRandom Forest (100 trees):")
    print(f"  Train Accuracy: {forest.score(X_train, y_train):.3f}")
    print(f"  Test Accuracy: {forest.score(X_test, y_test):.3f}")
    
    # Visualize prediction confidence
    if X.shape[1] == 2:  # Only if 2D
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Decision boundaries
        for ax, model, title in [(ax1, single_tree, 'Single Tree'),
                                  (ax2, forest, 'Random Forest')]:
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)
            
            cs = ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                      cmap=plt.cm.RdYlBu, edgecolor='black', s=50)
            ax.set_title(title)
            
        plt.tight_layout()
        plt.show()

demonstrate_random_forest()
```

## Advantages and Disadvantages 📊

### Advantages ✅
1. **Interpretable** - You can visualize and understand the decisions
2. **No scaling needed** - Trees don't care about feature scales
3. **Handles non-linearity** - Can capture complex patterns
4. **Feature importance** - Tells you what matters
5. **Handles mixed data** - Numerical and categorical together
6. **Fast predictions** - Just traverse the tree

### Disadvantages ❌
1. **Overfitting** - Trees love to memorize data
2. **Instability** - Small data changes → different trees
3. **Biased to dominant classes** - In imbalanced datasets
4. **Poor extrapolation** - Can't predict outside training range
5. **Axis-aligned splits** - Creates boxy boundaries

## Practical Tips 💡

### 1. Start Simple
```python
# Always start with a shallow tree
tree = DecisionTreeClassifier(max_depth=3)
# Then increase depth if needed
```

### 2. Use Cross-Validation for Hyperparameters
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

### 3. Handle Imbalanced Data
```python
# Use class weights
tree = DecisionTreeClassifier(class_weight='balanced')

# Or specify custom weights
weights = {0: 1, 1: 10}  # Make class 1 ten times more important
tree = DecisionTreeClassifier(class_weight=weights)
```

### 4. Visualize Your Trees
```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, fontsize=10)
plt.show()
```

### 5. Consider Ensembles
When single trees aren't enough:
- **Random Forest**: Many trees with random feature subsets
- **Extra Trees**: Even more randomness
- **Gradient Boosting**: Trees that fix each other's mistakes
- **XGBoost/LightGBM**: Industrial-strength boosting

## When to Use Decision Trees? 🤔

**Perfect for:**
- When interpretability is crucial
- Mixed numerical/categorical features
- Non-linear relationships
- Feature importance analysis
- Quick baseline models

**Avoid when:**
- Data has linear relationships (use linear models)
- Very high-dimensional sparse data (use linear SVM)
- Need probability calibration (trees are overconfident)
- Extrapolation is needed

## The Tree Philosophy 🌳

Decision trees embody a beautiful principle: complex decisions can be broken down into simple questions. They're like a wise mentor who teaches by asking the right questions rather than giving direct answers.

Key takeaways:
- Trees learn by recursively splitting data
- Information gain guides the splitting process
- Pruning prevents overfitting
- Random Forests combine many trees for stability
- Interpretability is their superpower

Remember: Sometimes the best model isn't the most accurate one - it's the one you can understand and trust!

---

*"The best time to plant a tree was 20 years ago. The second best time is now."* - Chinese Proverb

*"The best time to train a decision tree is when you need interpretable predictions!"* - ML Proverb

Happy tree growing! 🌱