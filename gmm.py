# Import libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split



# Compute BIC and AIC scores for model selection
def compute_bic_aic(log_likelihood, n_params, n_samples):
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    aic = -2 * log_likelihood + 2 * n_params
    return bic, aic

# Evaluate multivariate normal PDF at a given point
def MultiVarNormal(x, mean, cov):
    d = x.shape[0]
    # Regularize covariance to avoid singularity
    cov_reg = cov + np.eye(d) * 1e-6
    # Compute log-determinant for numerical stability
    sign, logdet = np.linalg.slogdet(cov_reg)
    # Log normalizing constant: -0.5 * (d*log(2π) + logdet)
    log_norm_const = -0.5 * (d * np.log(2 * np.pi) + logdet)
    # Quadratic form in the exponent
    quad_form = -0.5 * (x - mean).dot(np.linalg.pinv(cov_reg)).dot(x - mean)
    # Return PDF from log-domain
    return np.exp(log_norm_const + quad_form)

# Update mixture proportions (π) in the EM algorithm
def UpdateMixProps(hidden_matrix):
    n,k = hidden_matrix.shape
    mix_props = np.zeros(k)
    # begin solution
    mix_props = np.mean(hidden_matrix, axis=0)
    # end solution
    return mix_props

# Update the means (μ) in the EM algorithm
def UpdateMeans(X, hidden_matrix):
    n,d = X.shape
    k = hidden_matrix.shape[1]
    new_means = np.zeros([k,d])
    # begin solution
    for i in range(k):
        new_means[i, :] = (X.T * hidden_matrix[:, i]).T.sum(axis=0) / hidden_matrix[:, i].sum()
    # end solution
    return new_means

# Update covariance matrices (Σ) in the EM algorithm
def UpdateCovars(X, hidden_matrix, means):
    n, d = X.shape
    k = hidden_matrix.shape[1]
    new_covs = np.zeros((k, d, d))
    for i in range(k):
        weights = hidden_matrix[:, i] + 1e-10  # (n,)
        weight_sum = np.sum(weights)
        diff = X - means[i]  # (n, d)
        weighted_diff = diff.T * weights  # (d, n)
        cov_i = weighted_diff @ diff / weight_sum  # (d, d)
        cov_i += np.eye(d) * 1e-6  # regularize
        new_covs[i] = cov_i
    return new_covs

# E-step: compute responsibilities and log-likelihood
def HiddenMatrix(X, means, covs, mix_props):
    n, d = X.shape
    k = means.shape[0]
    # Preallocate log-probabilities
    log_t = np.zeros((n, k))
    # Compute log P(x|j) + log pi_j for each component j using full covariance matrices
    for j in range(k):
        cov_j = covs[j]
        sign, logdet = np.linalg.slogdet(cov_j)
        log_norm = -0.5 * (d * np.log(2 * np.pi) + logdet)
        inv_cov = np.linalg.pinv(cov_j)
        diff = X - means[j]
        quad = -0.5 * np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
        log_mix = np.log(np.maximum(mix_props[j], 1e-10))
        log_t[:, j] = log_norm + quad + log_mix
    
    # Convert log-probabilities into responsibilities with log-sum-exp
    hidden_matrix = np.zeros((n, k))
    ll = 0.0
    for i in range(n):
        li = log_t[i]                   # (k,)
        m = np.max(li)
        exp_li = np.exp(li - m)         # (k,)
        sum_exp = np.sum(exp_li)
        hidden_matrix[i] = exp_li / sum_exp
        ll += m + np.log(sum_exp)
    return hidden_matrix, ll

# Run EM algorithm for unsupervised GMM clustering
def GMM1(X, init_means, init_covs, init_mix_props, thres=0.001):
    n,d = X.shape
    k = init_means.shape[0]
    clusters = np.zeros(n)
    ll_list = []
    # begin solution
    means = init_means
    covs = init_covs
    mix_props = init_mix_props
    for i_itr in range(10000):
        #print(f'Start iteration {i_itr}')
        # Compute
        hidden_matrix,ll= HiddenMatrix(X, means, covs, mix_props)
        ll_list.append(ll)
        mix_props = UpdateMixProps(hidden_matrix)
        means = UpdateMeans(X, hidden_matrix)
        covs = UpdateCovars(X, hidden_matrix, means)  
        #print(f'Likelihoood at iteration {i_itr}: {ll_list[-1]}')      
        if len(ll_list)<=1:
            continue
        if ll_list[-1] - ll_list[-2] < thres:
            #print('Converged', ll_list[-1] - ll_list[-2])
            break
    clusters = np.argmax(hidden_matrix, axis=1) 
    # end solution
    return clusters,hidden_matrix,ll_list

# Perform PCA using SVD and return projection
def PCA(X, n_components):
    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)

    # Step 2: Compute SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    total_variance = np.sum(S**2)

    # Step 3: Project onto the top principal components
    X_pca = X_centered @ Vt[:n_components].T
    projection_matrix = Vt[:n_components, :]

    explained_variance = S**2 / np.sum(S**2)

    return X_pca, explained_variance, projection_matrix
    
# Estimate GMM parameters via supervised MLE (one Gaussian per class)
def GMM2(X, y_labels):
    unique_labels = np.unique(y_labels)
    k = len(unique_labels)
    n, d = X.shape
    means = np.zeros((k, d))
    covs = np.zeros((k, d, d))
    mix_props = np.zeros(k)
    for idx, lbl in enumerate(unique_labels):
        Xk = X[y_labels == lbl]
        nk = Xk.shape[0]
        mix_props[idx] = nk / n
        means[idx] = np.mean(Xk, axis=0)
        covs[idx] = np.cov(Xk.T) + np.eye(d) * 1e-6
    return means, covs, mix_props

# Estimate GMM with multiple Gaussians per class (semi-supervised)
def GMM3(X, y_labels, n_components_per_class=3, use_kmeans=False):
    unique_labels = np.unique(y_labels)
    d = X.shape[1]
    means_list, covs_list, mix_props_list = [], [], []

    for label in unique_labels:
        X_class = X[y_labels == label]
        n_class = X_class.shape[0]

        # Initialize parameters for EM using KMeans-based initialization
        if use_kmeans:
            init_means = KMeans(n_clusters=n_components_per_class, n_init=10, random_state=0).fit(X_class).cluster_centers_
        else:
            init_means = X_class[np.random.choice(n_class, n_components_per_class, replace=False)]

        init_cov = np.cov(X_class.T) + np.eye(d) * 1e-6
        init_covs = np.array([init_cov for _ in range(n_components_per_class)])
        init_mix = np.ones(n_components_per_class) / n_components_per_class

        _, hidden_class, _ = GMM1(X_class, init_means, init_covs, init_mix)

        # Update parameters using M-step formulas
        mix_props = UpdateMixProps(hidden_class) * (n_class / X.shape[0])  # adjust to global weight
        means = UpdateMeans(X_class, hidden_class)
        covs = UpdateCovars(X_class, hidden_class, means)

        means_list.append(means)
        covs_list.append(covs)
        mix_props_list.append(mix_props)

    global_means = np.vstack(means_list)
    global_covs = np.vstack(covs_list)
    global_mix_props = np.concatenate(mix_props_list)

    return global_means, global_covs, global_mix_props

# Evaluate GMMs with varying k using BIC/AIC
def evaluate_em_gmm_model_selection(X, max_components=10):
    n_samples, n_features = X.shape
    aics = []
    bics = []
    ks = range(1, max_components + 1)
    for k in ks:
        print(f"Evaluating GMM with {k} components...")
        init_means = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).cluster_centers_
        init_cov = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
        init_covs = np.array([init_cov for _ in range(k)])
        init_mix = np.ones(k) / k
        _, _, ll_list = GMM1(X, init_means, init_covs, init_mix)
        final_ll = ll_list[-1]
        n_params = (k - 1) + k * n_features + k * n_features * (n_features + 1) / 2
        bic, aic = compute_bic_aic(final_ll, n_params, n_samples)
        aics.append(aic)
        bics.append(bic)
    return ks, aics, bics

# Load dataset and split into train/test sets
def load_and_split_data(filepath, test_size=0.2, random_state=42):
    data = pd.read_csv(filepath, sep='\t')
    labels = data.iloc[:, 1].values
    X = data.iloc[:, 2:].values
    np.random.seed(random_state)
    idx = np.random.permutation(len(X))
    split = int((1 - test_size) * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    return X_train, X_test, y_train, y_test

# Standardize features and apply PCA
def standardize_and_pca(X_train, X_test, n_components):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1e-8
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    X_train_pca, var, proj = PCA(X_train_scaled, n_components)
    X_test_centered = X_test_scaled - np.mean(X_train_scaled, axis=0)
    X_test_pca = X_test_centered @ proj.T

    cumulative_variance = np.cumsum(var)
    num_components_90 = np.searchsorted(cumulative_variance, 0.90) + 1
    # print(f"Number of top PCs that explain 90% variance: {num_components_90}")

    return X_train_pca, X_test_pca

# Align predicted clusters with ground-truth labels using Hungarian algorithm
def align_labels(true_labels, pred_labels):
    le = LabelEncoder()
    true_encoded = le.fit_transform(true_labels)
    D = confusion_matrix(true_encoded, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-D)
    mapping = dict(zip(col_ind, row_ind))
    aligned_preds = np.array([mapping[p] if p in mapping else -1 for p in pred_labels])
    return aligned_preds

import matplotlib.ticker as ticker

# Plot and save confusion matrix
def plot_confusion_matrix(cm, class_names, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(cax)

    n_true, n_pred = cm.shape
    ax.set_xticks(np.arange(n_pred))
    ax.set_yticks(np.arange(n_true))
    ax.set_xticklabels([str(i) for i in range(n_pred)], ha="right")
    ax.set_yticklabels([str(i) for i in range(n_true)] if len(class_names) != n_true else class_names)

    # Annotate each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Evaluate clustering with ARI, NMI, silhouette, and F1
def evaluate_clusters(name, X, true_labels, pred_labels, return_f1=False, return_per_class_f1=False):
    le = LabelEncoder()
    y_enc = le.fit_transform(true_labels)
    pred_labels_aligned = align_labels(true_labels, pred_labels)
    ari = adjusted_rand_score(y_enc, pred_labels_aligned)
    nmi = normalized_mutual_info_score(y_enc, pred_labels_aligned)
    n_clusters = len(np.unique(pred_labels_aligned))
    if n_clusters <= 1:
        sil = float('nan')
    else:
        sil = silhouette_score(X, pred_labels_aligned)
    f1 = f1_score(y_enc, pred_labels_aligned, average='weighted')
    print(f"[{name}] ARI: {ari:.4f}, NMI: {nmi:.4f}, Silhouette: {sil:.4f}, F1: {f1:.4f}")
    # cm = confusion_matrix(y_enc, pred_labels_aligned)
    # plot_confusion_matrix(cm, class_names=np.unique(y_enc),
    #                       title=f"{name}: Confusion Matrix",
    #                       filename=f"{name.replace(' ', '_')}_confusion_matrix.png")
    
    if return_f1:
        return f1
        
    if return_per_class_f1:
        per_class_f1 = f1_score(y_enc, pred_labels_aligned, average=None)
        return per_class_f1, le.classes_

# Evaluate models across different PCA component counts
def evaluate_models_across_pcs(X_train_raw, X_test_raw, y_train, y_test, component_list=[10, 30, 50, 100, 150, 200, 244]):
    em_f1s, mle_f1s, gmm3_f1s = [], [], []
    for n_components in component_list:
        print(f"\n=== Evaluating with {n_components} Principal Components ===")
        X_train_pca, X_test_pca = standardize_and_pca(X_train_raw, X_test_raw, n_components=n_components)

        # EM-based GMM
        k = len(np.unique(y_train))
        init_means = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X_train_pca).cluster_centers_
        init_cov = np.cov(X_train_pca.T) + np.eye(X_train_pca.shape[1]) * 1e-6
        init_covs = np.array([init_cov for _ in range(k)])
        init_mix = np.ones(k) / k
        clusters_em, _, _ = GMM1(X_train_pca, init_means, init_covs, init_mix)
        evaluate_clusters(f"EM {n_components} PCs", X_train_pca, y_train, clusters_em)
        hidden_em_test, _ = HiddenMatrix(X_test_pca, init_means, init_covs, init_mix)
        clusters_em_test = np.argmax(hidden_em_test, axis=1)
        f1_em = evaluate_clusters(f"EM {n_components} PCs Test", X_test_pca, y_test, clusters_em_test, return_f1=True)
        em_f1s.append(f1_em)

        # MLE-based GMM
        mle_means, mle_covs, mle_mix = GMM2(X_train_pca, y_train)
        hidden_mle, _ = HiddenMatrix(X_train_pca, mle_means, mle_covs, mle_mix)
        clusters_mle = np.argmax(hidden_mle, axis=1)
        evaluate_clusters(f"MLE {n_components} PCs", X_train_pca, y_train, clusters_mle)
        hidden_mle_test, _ = HiddenMatrix(X_test_pca, mle_means, mle_covs, mle_mix)
        clusters_mle_test = np.argmax(hidden_mle_test, axis=1)
        f1_mle = evaluate_clusters(f"MLE {n_components} PCs Test", X_test_pca, y_test, clusters_mle_test, return_f1=True)
        mle_f1s.append(f1_mle)

        # Supervised GMM3
        gmm3_means, gmm3_covs, gmm3_mix = GMM3(X_train_pca, y_train, 2, use_kmeans=False)
        hidden_gmm3, _ = HiddenMatrix(X_train_pca, gmm3_means, gmm3_covs, gmm3_mix)
        clusters_gmm3 = np.argmax(hidden_gmm3, axis=1)
        evaluate_clusters(f"GMM3 {n_components} PCs", X_train_pca, y_train, clusters_gmm3)
        hidden_gmm3_test, _ = HiddenMatrix(X_test_pca, gmm3_means, gmm3_covs, gmm3_mix)
        clusters_gmm3_test = np.argmax(hidden_gmm3_test, axis=1)
        f1_gmm3 = evaluate_clusters(f"GMM3 {n_components} PCs Test", X_test_pca, y_test, clusters_gmm3_test, return_f1=True)
        gmm3_f1s.append(f1_gmm3)

        # f1_em_pc, labels = evaluate_clusters(f"EM {n_components} PCs Test", X_test_pca, y_test, clusters_em_test, return_per_class_f1=True)
        # f1_mle_pc, _ = evaluate_clusters(f"MLE {n_components} PCs Test", X_test_pca, y_test, clusters_mle_test, return_per_class_f1=True)
        # f1_gmm3_pc, _ = evaluate_clusters(f"GMM3 {n_components} PCs Test", X_test_pca, y_test, clusters_gmm3_test, return_per_class_f1=True)
        # # Ensure F1 score vectors match label length
        # min_len = min(len(labels), len(f1_gmm3_pc))
        # f1_em_pc = f1_em_pc[:min_len]
        # f1_mle_pc = f1_mle_pc[:min_len]
        # f1_gmm3_pc = f1_gmm3_pc[:min_len]
        # labels = labels[:min_len]

        # plt.figure(figsize=(8, 5))
        # plt.bar(labels, f1_em_pc, color='#9DC3E7')
        # plt.ylim(0, 1)
        # plt.ylabel('F1 Score')
        # plt.title(f'EM GMM Per-Class F1 Score - {n_components} PCs')
        # plt.tight_layout()
        # plt.savefig(f'f1_per_class_EM_GMM_{n_components}_pcs.png', dpi, bbox_inches='tight')
        # plt.close()

        # plt.figure(figsize=(8, 5))
        # plt.bar(labels, f1_mle_pc, color='#9DC3E7')
        # plt.ylim(0, 1)
        # plt.ylabel('F1 Score')
        # plt.title(f'MLE GMM Per-Class F1 Score - {n_components} PCs')
        # plt.tight_layout()
        # plt.savefig(f'f1_per_class_MLE_GMM_{n_components}_pcs.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # plt.figure(figsize=(8, 5))
        # plt.bar(labels, f1_gmm3_pc, color='#9DC3E7')
        # plt.ylim(0, 1)
        # plt.ylabel('F1 Score')
        # plt.title(f'GMM3 Per-Class F1 Score - {n_components} PCs')
        # plt.tight_layout()
        # plt.savefig(f'f1_per_class_GMM3_{n_components}_pcs.png', dpi=300, bbox_inches='tight')
        # plt.close()

    return em_f1s, mle_f1s, gmm3_f1s

# Perform k-fold cross-validation and test set evaluation
def cross_validate_and_evaluate(X, y, n_splits=5, n_components=50, val_size=0.2):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=42
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    em_models, mle_models, gmm3_models = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
        print(f"\n=== Fold {fold+1} ===")
        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        X_train_pca, X_val_pca = standardize_and_pca(X_train, X_val, n_components=n_components)

        # GMM1 - EM
        k = len(np.unique(y_train))
        init_means = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X_train_pca).cluster_centers_
        init_cov = np.cov(X_train_pca.T) + np.eye(X_train_pca.shape[1]) * 1e-6
        init_covs = np.array([init_cov for _ in range(k)])
        init_mix = np.ones(k) / k
        clusters_em_train, hidden_em_train, _ = GMM1(X_train_pca, init_means, init_covs, init_mix)
        mix_props_em = UpdateMixProps(hidden_em_train)
        means_em = UpdateMeans(X_train_pca, hidden_em_train)
        covs_em = UpdateCovars(X_train_pca, hidden_em_train, means_em)
        em_models.append((means_em, covs_em, mix_props_em, X_train))

        # GMM2 - MLE
        mle_means, mle_covs, mle_mix = GMM2(X_train_pca, y_train)
        mle_models.append((mle_means, mle_covs, mle_mix, X_train))

        # GMM3 - Per-Class Mixture
        gmm3_means, gmm3_covs, gmm3_mix = GMM3(X_train_pca, y_train, n_components_per_class=2)
        gmm3_models.append((gmm3_means, gmm3_covs, gmm3_mix, X_train))

        print(f"Trained GMM models on Fold {fold+1}")

    # Final Test Evaluation
    def evaluate_on_test(best_model, X_train_model, X_test, is_supervised):
        X_train_pca, X_test_pca = standardize_and_pca(X_train_model, X_test, n_components=n_components)
        means, covs, mix_props = best_model

        hidden_test, _ = HiddenMatrix(X_test_pca, means, covs, mix_props)
        clusters_test = np.argmax(hidden_test, axis=1)

        if is_supervised:
            clusters_test_aligned = align_labels(y_test, clusters_test)
        else:
            clusters_test_aligned = clusters_test

        le = LabelEncoder()
        y_true_enc = le.fit_transform(y_test)

        acc = accuracy_score(y_true_enc, clusters_test_aligned)
        precision = precision_score(y_true_enc, clusters_test_aligned, average='weighted', zero_division=0)
        recall = recall_score(y_true_enc, clusters_test_aligned, average='weighted', zero_division=0)
        f1 = f1_score(y_true_enc, clusters_test_aligned, average='weighted')
        return acc, precision, recall, f1

    def evaluate_model_list(model_list, is_supervised):
        results_per_fold = []
        for model in model_list:
            acc, precision, recall, f1 = evaluate_on_test(model[:3], model[3], X_test, is_supervised)
            results_per_fold.append((acc, precision, recall, f1))
        return results_per_fold

    results_em = evaluate_model_list(em_models, is_supervised=False)
    results_mle = evaluate_model_list(mle_models, is_supervised=True)
    results_gmm3 = evaluate_model_list(gmm3_models, is_supervised=True)

    return results_em, results_mle, results_gmm3

def main():
    np.random.seed(100)
    # Load and preprocess data
    X_train_raw, X_test_raw, y_train, y_test = load_and_split_data('data/LUAD_v12_20210228.tsv')
    # Extract only numeric features (columns 2228:24008)
    X_train_raw = X_train_raw[:, 2228:24008]
    X_test_raw = X_test_raw[:, 2228:24008]
    X_train_pca, X_test_pca = standardize_and_pca(X_train_raw, X_test_raw, n_components=10)

    # PCA evaluation
    X_train_scaled, _ = standardize_and_pca(X_train_raw, X_test_raw, n_components=X_train_raw.shape[1])[:2]
    X_centered = X_train_scaled - np.mean(X_train_scaled, axis=0)
    _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
    explained_variance = S**2 / np.sum(S**2)
    cumulative_variance = np.cumsum(explained_variance)
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='#9DC3E7')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs Number of PCs')
    #plt.grid(True)
    plt.axhline(y=0.9, color='red', linestyle='--', label='90% Variance')
    pc_target = 10
    var_at_pc_target = cumulative_variance[pc_target-1]  # pc_target-1 because Python is 0-indexed
    plt.axvline(x=pc_target, color='pink', linestyle='--', label=f'PC {pc_target}: {var_at_pc_target:.2f} variance')
    plt.annotate(f'{var_at_pc_target:.2%} variance',
                xy=(pc_target, var_at_pc_target),
                xytext=(pc_target+5, var_at_pc_target-0.1),
                arrowprops=dict(arrowstyle="->", color='pink'),
                fontsize=12)

    plt.legend()
    plt.tight_layout()
    plt.savefig('cumulative_variance_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # EM-based GMM
    k = len(np.unique(y_train))
    init_means = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_train_pca).cluster_centers_
    init_cov = np.cov(X_train_pca.T) + np.eye(X_train_pca.shape[1]) * 1e-6
    init_covs = np.array([init_cov for _ in range(k)])
    init_mix = np.ones(k) / k
    clusters_em, _, ll_em = GMM1(X_train_pca, init_means, init_covs, init_mix)
    evaluate_clusters("EM Train", X_train_pca, y_train, clusters_em)
    final_ll = ll_em
    n_samples, n_features = X_train_pca.shape
    n_components = k
    n_params = (n_components - 1) + n_components * n_features + n_components * n_features * (n_features + 1) / 2
    bic, aic = compute_bic_aic(final_ll[-1], n_params, n_samples)
    print(f"BIC: {bic:.2f}, AIC: {aic:.2f}")

    # Test evaluation for EM-based GMM
    hidden_em_test, _ = HiddenMatrix(X_test_pca, init_means, init_covs, init_mix)
    clusters_em_test = np.argmax(hidden_em_test, axis=1)
    evaluate_clusters("EM Test", X_test_pca, y_test, clusters_em_test)
    
    ks, aics, bics = evaluate_em_gmm_model_selection(X_train_pca, max_components=10)
    plt.figure(figsize=(8, 5))
    plt.plot(ks, aics, label='AIC', color='#EF8A43')
    plt.plot(ks, bics, label='BIC', color='#4865A9')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.title('AIC/BIC vs Number of GMM Components')
    plt.legend()
    plt.savefig("gmm_model_selection.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Supervised MLE GMM
    mle_means, mle_covs, mle_mix = GMM2(X_train_pca, y_train)
    hidden_mle, _ = HiddenMatrix(X_train_pca, mle_means, mle_covs, mle_mix)
    clusters_mle = np.argmax(hidden_mle, axis=1)
    evaluate_clusters("MLE Train", X_train_pca, y_train, clusters_mle)

    # Test evaluation
    hidden_mle_test, _ = HiddenMatrix(X_test_pca, mle_means, mle_covs, mle_mix)
    clusters_mle_test = np.argmax(hidden_mle_test, axis=1)
    evaluate_clusters("MLE Test", X_test_pca, y_test, clusters_mle_test)

    # Supervised Mixture-Per-Class GMM
    # With random initialization
    gmm3_means, gmm3_covs, gmm3_mix = GMM3(X_train_pca, y_train, 2)
    # gmm3_means, gmm3_covs, gmm3_mix = GMM3(X_train_pca, y_train, use_kmeans=True)

    hidden_gmm3, _ = HiddenMatrix(X_train_pca, gmm3_means, gmm3_covs, gmm3_mix)
    clusters_gmm3 = np.argmax(hidden_gmm3, axis=1)
    evaluate_clusters("MLE Mixture Per Class Train", X_train_pca, y_train, clusters_gmm3)

    # Test evaluation for Supervised Mixture-Per-Class GMM
    hidden_gmm3_test, _ = HiddenMatrix(X_test_pca, gmm3_means, gmm3_covs, gmm3_mix)
    clusters_gmm3_test = np.argmax(hidden_gmm3_test, axis=1)
    evaluate_clusters("MLE Mixture Per Class Test", X_test_pca, y_test, clusters_gmm3_test)

    component_list = [v for v in range(1, 13)]
    component_list += [v for v in range(15, 45, 5)]
    #component_list += [50, 100, 150, 200, 250]
    print(component_list)
    em_f1s, mle_f1s, gmm3_f1s = evaluate_models_across_pcs(X_train_raw, X_test_raw, y_train, y_test, component_list)
    plt.figure(figsize=(8, 5))
    plt.plot(component_list, em_f1s, marker='o', label='GMM1', color='#B2B2B2')
    plt.plot(component_list, mle_f1s, marker='s', label='GMM2', color='#D69C9B')
    plt.plot(component_list, gmm3_f1s, marker='^', label='GMM3', color='#74A0A1')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Test F1 Score')
    plt.title('Test F1 Score vs Number of PCs')
    xticks = [v for v in range(5, 45, 5)]
    plt.xticks(xticks)
    plt.xlim(0, max(xticks)+2)
    plt.axvline(x=10, linestyle='--', color='pink', linewidth=1)
    plt.legend()
    plt.savefig('test_f1_vs_pcs.png', dpi=300, bbox_inches='tight') 
    plt.close()

    print("\n=== Cross Validation and Final Evaluation ===")
    X_all = np.vstack([X_train_raw, X_test_raw])
    y_all = np.concatenate([y_train, y_test])
    results_em, results_mle, results_gmm3 = cross_validate_and_evaluate(X_all, y_all, n_splits=5, n_components=8)

    print("\n=== Final Evaluation on Held-Out Test Set ===")

    for i, (res_em, res_mle, res_gmm3) in enumerate(zip(results_em, results_mle, results_gmm3), 1):
        print(f"\n--- Fold {i} ---")
        print(f"GMM1 (EM): Accuracy={res_em[0]:.4f}, Precision={res_em[1]:.4f}, Recall={res_em[2]:.4f}, F1={res_em[3]:.4f}")
        print(f"GMM2 (MLE): Accuracy={res_mle[0]:.4f}, Precision={res_mle[1]:.4f}, Recall={res_mle[2]:.4f}, F1={res_mle[3]:.4f}")
        print(f"GMM3 (Supervised Mixture): Accuracy={res_gmm3[0]:.4f}, Precision={res_gmm3[1]:.4f}, Recall={res_gmm3[2]:.4f}, F1={res_gmm3[3]:.4f}")


    # print("\n=== Training on Raw Normalized Data (No PCA) ===")
    # # Standardize raw data
    # X_train_mean = np.mean(X_train_raw, axis=0)
    # X_train_std = np.std(X_train_raw, axis=0)
    # X_train_std[X_train_std == 0] = 1e-8
    # X_train_scaled = (X_train_raw - X_train_mean) / X_train_std
    # X_test_scaled = (X_test_raw - X_train_mean) / X_train_std

    # # EM-based GMM on raw data
    # init_means_raw = X_train_scaled[np.random.choice(X_train_scaled.shape[0], k, replace=False)]
    # init_cov_raw = np.cov(X_train_scaled.T) + np.eye(X_train_scaled.shape[1]) * 1e-6
    # init_covs_raw = np.array([init_cov_raw for _ in range(k)])
    # init_mix_raw = np.ones(k) / k
    # clusters_em_raw, _, _ = GMM1(X_train_scaled, init_means_raw, init_covs_raw, init_mix_raw)
    # evaluate_clusters("EM Raw", X_train_scaled, y_train, clusters_em_raw)

    # # Supervised MLE GMM on raw data
    # mle_means_raw, mle_covs_raw, mle_mix_raw = GMM2(X_train_scaled, y_train)
    # hidden_mle_raw, _ = HiddenMatrix(X_train_scaled, mle_means_raw, mle_covs_raw, mle_mix_raw)
    # clusters_mle_raw = np.argmax(hidden_mle_raw, axis=1)
    # evaluate_clusters("MLE Train Raw", X_train_scaled, y_train, clusters_mle_raw)

    # # Test evaluation on raw data
    # hidden_mle_test_raw, _ = HiddenMatrix(X_test_scaled, mle_means_raw, mle_covs_raw, mle_mix_raw)
    # clusters_mle_test_raw = np.argmax(hidden_mle_test_raw, axis=1)
    # evaluate_clusters("MLE Test Raw", X_test_scaled, y_test, clusters_mle_test_raw)

if __name__ == "__main__":
    main()