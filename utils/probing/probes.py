from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset


def get_sample_weights(y_train):
    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))
    # Create sample weights based on class weights
    sample_weights = np.array([class_weight_dict[y] for y in y_train])
    return sample_weights


def remove_identical_points(activations, labels):

    if isinstance(labels, list):
        labels = np.array(labels)

    _, base_idx, _ = np.unique(
        activations, axis=0, return_index=True, return_inverse=True
    )
    indices = np.sort(base_idx)
    activations = activations[indices]
    labels = labels[indices]
    return activations, labels


def fit_sklearn(X_train, y_train, X_test, y_test):

    X_train, y_train = remove_identical_points(X_train, y_train)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sample_weights = get_sample_weights(y_train)

    param_grid = {"C": [0.001, 0.01, 0.1]}
    probe = LogisticRegression
    max_iter = 2000
    params = {"max_iter": max_iter, "class_weight": None, "tol": 2e-4}
    fit_kwargs = {"sample_weight": sample_weights}

    grid_search = GridSearchCV(
        estimator=probe(**params),
        param_grid=param_grid,
        cv=skf,
        scoring="balanced_accuracy",  # or 'f1_macro', 'balanced_accuracy' etc.
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train, **fit_kwargs)
    best_params = grid_search.best_params_
    print(best_params)
    params.update(**best_params)

    final_model = probe(**params)
    final_model.fit(X_train, y_train, **fit_kwargs)

    X_test, y_test = remove_identical_points(X_test, y_test)
    # X_test = scaler.transform(X_test)

    y_pred = final_model.predict(X_test)
    test_acc = balanced_accuracy_score(y_test, y_pred)

    return test_acc


def fit_sklearn_no_cv(X_train, y_train, X_test, y_test):
    X_train, y_train = remove_identical_points(X_train, y_train)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)

    # Degenerate layer: every training sample produced an identical activation
    # row, so remove_identical_points collapsed them to one (and one label).
    # Happens at layer 0 when every sample ends in the same scaffold token id
    # under --use_last_token — LayerNorm(embedding(tok)) is deterministic per
    # token, so identical last tokens give identical activations. Return
    # chance-level balanced accuracy for this layer instead of crashing the
    # whole sweep.
    if len(np.unique(y_train)) < 2:
        n_classes = max(2, len(np.unique(y_test)))
        return 1.0 / n_classes

    model = LogisticRegression(max_iter=1000, C=0.01, tol=2e-4)
    model.fit(X_train, y_train, sample_weight=get_sample_weights(y_train))

    X_test, y_test = remove_identical_points(X_test, y_test)
    # X_test = scaler.transform(X_test)

    y_pred = model.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)


class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def fit_mlp(
    X_train,
    y_train,
    X_test,
    y_test,
    X_val,
    y_val,
    batch_size=256,
    epochs=2,
):

    # === Convert to Torch Tensors ===
    device = torch.device("cuda")

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # === Dataloaders ===
    batch_size = batch_size
    train_dataset = TensorDataset(X_train, y_train_tensor)
    val_dataset = TensorDataset(X_val, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # === Model Config ===
    input_dim = X_train.shape[1]
    hidden_dim = 200

    output_dim = len(set(y_train))
    num_epochs = epochs

    lrs = [1e-6, 1e-5, 1e-4]
    weight_decays = [1e-6, 1e-5, 1e-4, 1e-3]

    # === Search for Best Weight Decay ===
    best_val_acc = 0
    best_model_state = None
    perf = {}

    for lr, wd in product(weight_decays, lrs):
        model = TwoLayerNN(input_dim, hidden_dim, output_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        # Validation accuracy
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(yb)
        val_preds = torch.cat(all_preds)
        val_labels = torch.cat(all_labels)
        val_acc = balanced_accuracy_score(val_labels, val_preds)

        perf[f"lr{lr}_wd{wd}"] = val_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_params = f"lr{lr}_wd{wd}"

    # === Final Model Evaluation ===
    model = TwoLayerNN(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        test_preds = model(X_test).argmax(dim=1)
        test_acc = balanced_accuracy_score(y_test_tensor.cpu(), test_preds.cpu())

    return test_acc, perf, best_params
