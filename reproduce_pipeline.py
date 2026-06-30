"""
Reproducible pipeline for:
"Measurement Scarcity Limits Multi-Channel Cardiac Ion-Channel Liability
Prediction: A Public-Data Audit"

Runs the full analysis end to end:
  1. Load hERG (TDC) + Nav1.5 / Cav1.2 (ChEMBL)
  2. Clean & label (IC50 -> pIC50, threshold 6)
  3. Featurize (10 descriptors + 2048-bit Morgan fingerprint)
  4. Train hERG baselines (LogReg / RandomForest / XGBoost)
  5. Scarcity table + cross-channel overlap
  6. Data-starvation learning curve (controlled experiment)
  7. Real Nav1.5 / Cav1.2 models: random vs scaffold split, bootstrap 95% CIs

Tested on Google Colab (CPU). Versions pinned in requirements.txt.
Fixed seed = 42 throughout for reproducibility.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, matthews_corrcoef, f1_score,
                             balanced_accuracy_score)
from xgboost import XGBClassifier

from tdc.single_pred import Tox
from chembl_webresource_client.new_client import new_client

SEED = 42
DESC_NAMES = ["MolWt", "LogP", "TPSA", "HBA", "HBD", "RotatableBonds",
              "AromaticRings", "RingCount", "FractionCSP3", "HeavyAtoms"]


# ----------------------------------------------------------------------
# Featurization
# ----------------------------------------------------------------------
def physchem(mol):
    return [Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol), mol.GetNumHeavyAtoms()]


def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros((2048,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return physchem(mol) + arr.tolist()


def build_X_y(df, smiles_col, label_col):
    rows, labels = [], []
    for s, y in zip(df[smiles_col], df[label_col]):
        f = featurize(s)
        if f is not None:
            rows.append(f)
            labels.append(int(y))
    return np.array(rows), np.array(labels)


def canonical(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


# ----------------------------------------------------------------------
# Data loading + cleaning
# ----------------------------------------------------------------------
def load_herg():
    split = Tox(name="hERG_Karim").get_split()
    return split["train"], split["test"]


def pull_chembl(target_id):
    recs = new_client.activity.filter(
        target_chembl_id=target_id, standard_type="IC50"
    ).only(["molecule_chembl_id", "canonical_smiles",
            "standard_value", "standard_units"])
    return pd.DataFrame(recs)


def clean_channel(raw, name):
    df = raw.copy()
    df = df[df["standard_units"] == "nM"]
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])
    df = df[df["standard_value"] > 0]
    df["pIC50"] = 9 - np.log10(df["standard_value"])
    g = (df.groupby(["molecule_chembl_id", "canonical_smiles"])["pIC50"]
           .median().reset_index())
    g["Y"] = (g["pIC50"] >= 6).astype(int)
    g["channel"] = name
    print(f"{name}: {len(g)} unique | balance {np.bincount(g['Y'])}")
    return g


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------
def bootstrap_ci(y, proba, preds, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs, mccs = [], []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], proba[s]))
        mccs.append(matthews_corrcoef(y[s], preds[s]))
    pct = lambda a: (round(np.percentile(a, 2.5), 3), round(np.percentile(a, 97.5), 3))
    return np.median(aucs), pct(aucs), np.median(mccs), pct(mccs)


def scaffold_split_idx(smiles_list, test_frac=0.25, seed=SEED):
    groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else f"none{i}"
        groups[sc or f"none{i}"].append(i)
    gl = sorted(groups.values(), key=len, reverse=True)
    np.random.default_rng(seed).shuffle(gl)
    test, train, n = [], [], len(smiles_list)
    for g in gl:
        (test if len(test) < test_frac * n else train).extend(g)
    return np.array(train), np.array(test)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # ---- hERG baselines ----
    tr, te = load_herg()
    Xtr, ytr = build_X_y(tr, "Drug", "Y")
    Xte, yte = build_X_y(te, "Drug", "Y")

    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED).fit(sc.transform(Xtr), ytr)
    for name, model, Xt in [("LogReg", lr, sc.transform(Xte))]:
        p = model.predict_proba(Xt)[:, 1]
        print(name, "ROC-AUC", round(roc_auc_score(yte, p), 3))
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    print("RandomForest ROC-AUC", round(roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]), 3))
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.9, random_state=SEED, eval_metric="logloss",
                        n_jobs=-1).fit(Xtr, ytr)
    print("XGBoost ROC-AUC", round(roc_auc_score(yte, xgb.predict_proba(Xte)[:, 1]), 3))

    # ---- Nav1.5 / Cav1.2 ----
    nav = clean_channel(pull_chembl("CHEMBL1980"), "Nav1.5")   # SCN5A
    cav = clean_channel(pull_chembl("CHEMBL1940"), "Cav1.2")   # CACNA1C

    # ---- learning curve (controlled starvation) ----
    print("\nData-starvation learning curve (hERG, fixed test):")
    rng = np.random.default_rng(SEED)
    for size in [9412, 5000, 2830, 1500, 800, 463, 250, 120]:
        aucs = []
        for rep in range(5):
            idx = rng.choice(len(Xtr), size=min(size, len(Xtr)), replace=False)
            m = RandomForestClassifier(n_estimators=300, random_state=rep, n_jobs=-1)
            m.fit(Xtr[idx], ytr[idx])
            aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        print(f"  size {size:>5}: {np.mean(aucs):.3f} (+/-{np.std(aucs):.3f})")

    # ---- real channel models, random vs scaffold, with CIs ----
    for df in (nav, cav):
        name = df["channel"].iloc[0]
        X, y = build_X_y(df, "canonical_smiles", "Y")
        smi = df["canonical_smiles"].tolist()
        for split_name, (tri, tei) in [
            ("Random", train_test_split(np.arange(len(y)), test_size=0.25,
                                        stratify=y, random_state=SEED)),
            ("Scaffold", scaffold_split_idx(smi)),
        ]:
            # train_test_split returns arrays directly for Random; unify shape
            if split_name == "Random":
                tri, tei = train_test_split(np.arange(len(y)), test_size=0.25,
                                            stratify=y, random_state=SEED)
            m = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                       n_jobs=-1, class_weight="balanced")
            m.fit(X[tri], y[tri])
            proba = m.predict_proba(X[tei])[:, 1]
            preds = (proba >= 0.5).astype(int)
            auc, auc_ci, mcc, mcc_ci = bootstrap_ci(y[tei], proba, preds)
            print(f"{name} {split_name}: na={int(y[tei].sum())} "
                  f"ROC-AUC={auc:.3f} {auc_ci}  MCC={mcc:.3f} {mcc_ci}")


if __name__ == "__main__":
    main()
