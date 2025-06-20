---
layout: post
title: "Naive Bayes: The Power of Simple Assumptions"
date: 2025-06-20
categories: [machine-learning, classification, probability]
excerpt: "Discover how Naive Bayes uses probability and a 'naive' assumption to create surprisingly effective classifiers, especially for text classification and spam detection"
---

# Naive Bayes: The Power of Simple Assumptions 🎲

## The Big Idea: Think Like a Detective

Imagine you're a detective trying to figure out if an email is spam. You notice:
- It contains the word "FREE" (suspicious...)
- It has "CLICK HERE" (very suspicious...)
- It mentions "Nigerian prince" (case closed!)

Each clue adjusts your belief about whether it's spam. That's exactly how Naive Bayes works - it combines evidence using probability theory to make predictions.

## Bayes' Theorem: The Foundation 🧮

Everything starts with Bayes' theorem, a beautiful equation that tells us how to update our beliefs given new evidence:

```
P(A|B) = P(B|A) × P(A) / P(B)
```

Let me translate this to English:
- `P(A|B)` = Probability of A given that B happened (posterior)
- `P(B|A)` = Probability of B given that A is true (likelihood)
- `P(A)` = Probability of A (prior)
- `P(B)` = Probability of B (evidence)

### A Medical Example

Let's make this concrete. Suppose:
- A disease affects 1% of people: `P(Disease) = 0.01`
- A test is 90% accurate: `P(Positive|Disease) = 0.90`
- The test has 5% false positives: `P(Positive|No Disease) = 0.05`

If you test positive, what's the probability you have the disease?

```python
# Prior probabilities
p_disease = 0.01
p_no_disease = 0.99

# Likelihoods
p_positive_given_disease = 0.90
p_positive_given_no_disease = 0.05

# Total probability of testing positive
p_positive = (p_positive_given_disease * p_disease + 
              p_positive_given_no_disease * p_no_disease)

# Apply Bayes' theorem
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"P(Disease|Positive Test) = {p_disease_given_positive:.2%}")
# Result: 15.38% - Much lower than you might expect!
```

## The "Naive" Assumption: Why It's Brilliant 💡

Naive Bayes makes one key assumption: **all features are independent given the class**. 

This is "naive" because in reality, features often correlate. For text:
- "New" and "York" often appear together
- "Machine" and "Learning" are best friends

But here's the magic: even with this wrong assumption, Naive Bayes often works incredibly well! It's like assuming all your detective clues are independent - not true, but good enough to catch the criminal.

## Implementation: Building Naive Bayes from Scratch 🛠️

Let's implement three types of Naive Bayes:

### 1. Gaussian Naive Bayes (for continuous features)

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class GaussianNaiveBayes:
    """Naive Bayes for continuous features (assumes Gaussian distribution)"""
    
    def __init__(self):
        self.class_priors = {}
        self.feature_stats = {}  # Mean and std for each feature per class
        self.classes = None
    
    def fit(self, X, y):
        """Train the classifier"""
        self.classes = np.unique(y)
        n_samples = X.shape[0]
        
        for c in self.classes:
            # Calculate prior P(class)
            mask = (y == c)
            self.class_priors[c] = np.sum(mask) / n_samples
            
            # Calculate mean and std for each feature
            X_c = X[mask]
            self.feature_stats[c] = {
                'mean': np.mean(X_c, axis=0),
                'std': np.std(X_c, axis=0) + 1e-6  # Add small value to avoid division by zero
            }
    
    def _calculate_likelihood(self, x, mean, std):
        """Calculate Gaussian probability density"""
        # P(x|class) = 1/√(2πσ²) * exp(-(x-μ)²/2σ²)
        return norm.pdf(x, mean, std)
    
    def _predict_sample(self, x):
        """Predict class for a single sample"""
        posteriors = {}
        
        for c in self.classes:
            # Start with prior
            posterior = np.log(self.class_priors[c])
            
            # Multiply by likelihood for each feature
            # Use log to avoid numerical underflow
            for i, feature_val in enumerate(x):
                likelihood = self._calculate_likelihood(
                    feature_val,
                    self.feature_stats[c]['mean'][i],
                    self.feature_stats[c]['std'][i]
                )
                posterior += np.log(likelihood + 1e-10)
            
            posteriors[c] = posterior
        
        # Return class with highest posterior
        return max(posteriors, key=posteriors.get)
    
    def predict(self, X):
        """Predict classes for samples"""
        return np.array([self._predict_sample(x) for x in X])
    
    def predict_proba(self, X):
        """Get probability estimates"""
        probas = []
        
        for x in X:
            posteriors = {}
            
            for c in self.classes:
                posterior = np.log(self.class_priors[c])
                
                for i, feature_val in enumerate(x):
                    likelihood = self._calculate_likelihood(
                        feature_val,
                        self.feature_stats[c]['mean'][i],
                        self.feature_stats[c]['std'][i]
                    )
                    posterior += np.log(likelihood + 1e-10)
                
                posteriors[c] = posterior
            
            # Convert log probabilities back to probabilities
            max_log_prob = max(posteriors.values())
            exp_probs = {c: np.exp(p - max_log_prob) for c, p in posteriors.items()}
            total = sum(exp_probs.values())
            
            # Normalize
            probas.append([exp_probs[c] / total for c in self.classes])
        
        return np.array(probas)
    
    def visualize_distributions(self, X, y, feature_names=None):
        """Visualize the Gaussian distributions for each feature"""
        n_features = X.shape[1]
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
        
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4))
        if n_features == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            for c in self.classes:
                mask = (y == c)
                X_c = X[mask, i]
                
                # Plot histogram
                ax.hist(X_c, alpha=0.5, label=f'Class {c}', density=True, bins=20)
                
                # Plot fitted Gaussian
                x_range = np.linspace(X[:, i].min(), X[:, i].max(), 100)
                gaussian = norm.pdf(x_range, 
                                  self.feature_stats[c]['mean'][i],
                                  self.feature_stats[c]['std'][i])
                ax.plot(x_range, gaussian, linewidth=2)
            
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel('Density')
            ax.legend()
            ax.set_title(f'{feature_names[i]} Distribution by Class')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1,
                              class_sep=2.0, random_state=42)
    
    # Train our Naive Bayes
    gnb = GaussianNaiveBayes()
    gnb.fit(X, y)
    
    # Visualize distributions
    gnb.visualize_distributions(X, y, feature_names=['Feature 1', 'Feature 2'])
    
    # Make predictions
    predictions = gnb.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Training Accuracy: {accuracy:.3f}")
```

### 2. Multinomial Naive Bayes (for count data)

Perfect for text classification where features are word counts:

```python
class MultinomialNaiveBayes:
    """Naive Bayes for discrete count data (e.g., word counts)"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None
        self.n_features = None
    
    def fit(self, X, y):
        """Train on count data"""
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        n_samples = X.shape[0]
        
        for c in self.classes:
            mask = (y == c)
            X_c = X[mask]
            
            # Prior probability
            self.class_priors[c] = np.sum(mask) / n_samples
            
            # Feature probabilities with Laplace smoothing
            # P(feature_i | class_c) = (count_i + α) / (total_count + α * n_features)
            feature_counts = np.sum(X_c, axis=0)
            total_count = np.sum(feature_counts)
            
            self.feature_probs[c] = (feature_counts + self.alpha) / \
                                   (total_count + self.alpha * self.n_features)
    
    def predict(self, X):
        """Predict classes"""
        predictions = []
        
        for x in X:
            posteriors = {}
            
            for c in self.classes:
                # Log probability to avoid underflow
                log_posterior = np.log(self.class_priors[c])
                
                # Add log probabilities for present features
                for i, count in enumerate(x):
                    if count > 0:
                        log_posterior += count * np.log(self.feature_probs[c][i])
                
                posteriors[c] = log_posterior
            
            predictions.append(max(posteriors, key=posteriors.get))
        
        return np.array(predictions)
    
    def get_feature_log_prob(self):
        """Get log probabilities of features (useful for feature analysis)"""
        return {c: np.log(probs) for c, probs in self.feature_probs.items()}
```

### 3. Bernoulli Naive Bayes (for binary features)

Great for binary feature vectors (document contains word or not):

```python
class BernoulliNaiveBayes:
    """Naive Bayes for binary features"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None
    
    def fit(self, X, y):
        """Train on binary data"""
        self.classes = np.unique(y)
        n_samples = X.shape[0]
        
        for c in self.classes:
            mask = (y == c)
            X_c = X[mask]
            
            # Prior probability
            self.class_priors[c] = np.sum(mask) / n_samples
            
            # Feature probabilities: P(feature=1|class)
            # With Laplace smoothing
            feature_counts = np.sum(X_c, axis=0)
            n_samples_c = X_c.shape[0]
            
            self.feature_probs[c] = (feature_counts + self.alpha) / \
                                   (n_samples_c + 2 * self.alpha)
    
    def predict(self, X):
        """Predict classes"""
        predictions = []
        
        for x in X:
            posteriors = {}
            
            for c in self.classes:
                log_posterior = np.log(self.class_priors[c])
                
                for i, feature_val in enumerate(x):
                    if feature_val == 1:
                        # Feature is present
                        log_posterior += np.log(self.feature_probs[c][i])
                    else:
                        # Feature is absent
                        log_posterior += np.log(1 - self.feature_probs[c][i])
                
                posteriors[c] = log_posterior
            
            predictions.append(max(posteriors, key=posteriors.get))
        
        return np.array(predictions)
```

## Real-World Application: Spam Detection 📧

Let's build a spam classifier to see Naive Bayes in action:

```python
import re
from collections import Counter
from sklearn.model_selection import train_test_split

class SpamClassifier:
    """Simple spam classifier using Multinomial Naive Bayes"""
    
    def __init__(self):
        self.vocabulary = {}
        self.nb = MultinomialNaiveBayes()
    
    def preprocess_text(self, text):
        """Convert text to lowercase and extract words"""
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def build_vocabulary(self, texts):
        """Build vocabulary from training texts"""
        all_words = []
        for text in texts:
            all_words.extend(self.preprocess_text(text))
        
        # Keep most common words
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(1000)
        
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(common_words)}
    
    def text_to_features(self, text):
        """Convert text to feature vector"""
        words = self.preprocess_text(text)
        word_counts = Counter(words)
        
        # Create feature vector
        features = np.zeros(len(self.vocabulary))
        for word, count in word_counts.items():
            if word in self.vocabulary:
                features[self.vocabulary[word]] = count
        
        return features
    
    def fit(self, texts, labels):
        """Train the spam classifier"""
        # Build vocabulary
        self.build_vocabulary(texts)
        
        # Convert texts to features
        X = np.array([self.text_to_features(text) for text in texts])
        
        # Train Naive Bayes
        self.nb.fit(X, labels)
    
    def predict(self, texts):
        """Predict spam/ham for new texts"""
        X = np.array([self.text_to_features(text) for text in texts])
        return self.nb.predict(X)
    
    def get_spam_words(self, top_n=20):
        """Get words most indicative of spam"""
        if not hasattr(self.nb, 'feature_probs'):
            return []
        
        # Calculate log probability ratios
        spam_probs = self.nb.feature_probs[1]  # Assuming 1 = spam
        ham_probs = self.nb.feature_probs[0]   # Assuming 0 = ham
        
        log_ratios = np.log(spam_probs / ham_probs)
        
        # Get top spam indicators
        top_indices = np.argsort(log_ratios)[-top_n:][::-1]
        
        # Map back to words
        idx_to_word = {idx: word for word, idx in self.vocabulary.items()}
        spam_words = [(idx_to_word[idx], log_ratios[idx]) for idx in top_indices]
        
        return spam_words

# Example usage
if __name__ == "__main__":
    # Sample spam/ham emails
    emails = [
        # Ham (0)
        "Hey, want to grab lunch tomorrow?",
        "The meeting is scheduled for 3 PM in the conference room.",
        "Thanks for your help with the project yesterday!",
        "Can you review this document when you have time?",
        "Happy birthday! Hope you have a great day!",
        
        # Spam (1)
        "CONGRATULATIONS! You've won $1,000,000! Click here now!",
        "FREE VIAGRA! Best prices! Order now! Limited time offer!",
        "Make money fast! Work from home! Earn $5000 per week!",
        "Hot singles in your area! Click here to meet them!",
        "Nigerian prince needs your help. Send bank details for reward!"
    ]
    
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Train classifier
    spam_clf = SpamClassifier()
    spam_clf.fit(emails, labels)
    
    # Test on new emails
    test_emails = [
        "Let's discuss the quarterly report",
        "WIN FREE IPHONE! CLICK NOW! LIMITED OFFER!"
    ]
    
    predictions = spam_clf.predict(test_emails)
    for email, pred in zip(test_emails, predictions):
        label = "SPAM" if pred == 1 else "HAM"
        print(f"{label}: {email}")
    
    # Show most "spammy" words
    print("\nTop spam indicators:")
    for word, score in spam_clf.get_spam_words(10):
        print(f"  {word}: {score:.2f}")
```

## Visualizing Decision Boundaries 🎨

Let's see how Naive Bayes creates decision boundaries:

```python
def visualize_naive_bayes_boundaries():
    """Compare decision boundaries of different classifiers"""
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    
    # Generate 2D data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1,
                              class_sep=1.5, random_state=42)
    
    classifiers = [
        ('Naive Bayes', GaussianNB()),
        ('LDA', LinearDiscriminantAnalysis()),
        ('SVM (linear)', SVC(kernel='linear', probability=True)),
        ('SVM (RBF)', SVC(kernel='rbf', probability=True))
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (name, clf) in enumerate(classifiers):
        ax = axes[idx]
        
        # Train classifier
        clf.fit(X, y)
        
        # Create mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Get predictions
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
                  edgecolor='black', s=50)
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(f'{name}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()

visualize_naive_bayes_boundaries()
```

## The Math Behind Text Classification 📚

For text classification, we often use Multinomial Naive Bayes. Here's the math:

Given a document d with words w₁, w₂, ..., wₙ:

```
P(class|d) ∝ P(class) × ∏ P(wᵢ|class)^count(wᵢ)
```

In log space (to avoid underflow):
```
log P(class|d) = log P(class) + Σ count(wᵢ) × log P(wᵢ|class)
```

### Laplace Smoothing

To handle words not seen in training:
```
P(word|class) = (count(word, class) + α) / (total_words_in_class + α × vocabulary_size)
```

Where α (usually 1) prevents zero probabilities.

## Advantages and Disadvantages 📊

### Advantages ✅
1. **Fast training and prediction** - Just counting!
2. **Works with small datasets** - Needs less data than discriminative models
3. **Handles high dimensions well** - Great for text (thousands of features)
4. **Naturally multiclass** - No need for one-vs-rest tricks
5. **Interpretable** - Can see which features matter for each class
6. **Online learning** - Can update with new data easily

### Disadvantages ❌
1. **Strong independence assumption** - Often violated in practice
2. **Can't learn interactions** - "New York" treated as "New" + "York"
3. **Sensitive to data representation** - Feature engineering matters
4. **Zero frequency problem** - Needs smoothing for unseen features
5. **Calibration issues** - Probabilities often too extreme (near 0 or 1)

## Practical Tips 💡

### 1. Feature Engineering Matters
```python
# For text: Try different representations
# - Binary (word present/absent)
# - Counts (how many times)
# - TF-IDF (weighted by importance)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Binary features
binary_vectorizer = CountVectorizer(binary=True)

# Count features
count_vectorizer = CountVectorizer()

# TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
```

### 2. Handle Numerical Features Carefully
```python
# Option 1: Discretize continuous features
from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
X_discrete = discretizer.fit_transform(X_continuous)

# Option 2: Use Gaussian Naive Bayes
# But check if features are actually Gaussian!
```

### 3. Cross-Validation for Alpha
```python
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid_search.fit(X, y)
print(f"Best alpha: {grid_search.best_params_['alpha']}")
```

### 4. Complement Naive Bayes for Imbalanced Data
```python
from sklearn.naive_bayes import ComplementNB

# Better for imbalanced datasets
cnb = ComplementNB()
cnb.fit(X, y)
```

### 5. Feature Selection
```python
# Naive Bayes gives you feature probabilities - use them!
def select_features_by_class_correlation(X, y, nb_model, top_k=100):
    """Select features most correlated with classes"""
    # Get log probability ratios
    log_probs = nb_model.feature_log_prob_
    
    # Calculate variance across classes
    feature_variance = np.var(log_probs, axis=0)
    
    # Select top k most discriminative features
    top_features = np.argsort(feature_variance)[-top_k:]
    
    return X[:, top_features], top_features
```

## When to Use Naive Bayes? 🤔

**Perfect for:**
- Text classification (spam, sentiment, categories)
- Real-time predictions (it's fast!)
- High-dimensional data
- Small training sets
- Baseline models
- When interpretability matters

**Avoid when:**
- Features are strongly correlated
- You need to model feature interactions
- Probability calibration is crucial
- The independence assumption is badly violated

## Advanced Topic: Semi-Supervised Naive Bayes 🚀

When you have lots of unlabeled data:

```python
class SemiSupervisedNB:
    """Use unlabeled data to improve Naive Bayes"""
    
    def __init__(self, nb_model):
        self.nb = nb_model
    
    def fit(self, X_labeled, y_labeled, X_unlabeled, n_iterations=10):
        """EM algorithm for semi-supervised learning"""
        # Start with supervised learning
        self.nb.fit(X_labeled, y_labeled)
        
        for iteration in range(n_iterations):
            # E-step: Predict labels for unlabeled data
            y_pseudo = self.nb.predict(X_unlabeled)
            
            # M-step: Retrain with all data
            X_all = np.vstack([X_labeled, X_unlabeled])
            y_all = np.hstack([y_labeled, y_pseudo])
            
            self.nb.fit(X_all, y_all)
            
            print(f"Iteration {iteration + 1} complete")
    
    def predict(self, X):
        return self.nb.predict(X)
```

## Conclusion: The Elegance of Simplicity 🎯

Naive Bayes proves that sometimes the simplest ideas are the most powerful. By making a "naive" assumption, we get:
- Lightning-fast training
- Interpretable models
- Surprisingly good performance

It's like the Swiss Army knife of machine learning - not always the best tool for any specific job, but reliable and useful in many situations.

Key takeaways:
- **Bayes' theorem** provides the mathematical foundation
- **Independence assumption** makes computation tractable
- **Different variants** for different data types
- **Great for text** and high-dimensional data
- **Fast and interpretable** but sometimes overconfident

Remember: In machine learning, "naive" doesn't mean stupid - it means making smart simplifications!

---

*"It is better to be approximately right than precisely wrong."* - John Tukey

*"And Naive Bayes is often approximately right enough!"* - Every ML practitioner

Happy classifying! 🎲