#!/usr/bin/env python
"""
Benchmark4: Main Evaluation Script.

For each (dataset, descriptor) pair:
  1. Load PH cache (or images for image-based descriptors)
  2. Get optimal params from exp4 rules
  3. Extract features (grayscale or per-channel RGB)
  4. Run 5-fold stratified CV with all applicable classifiers
  5. Save JSON results

Usage:
    python benchmarks/benchmark4/evaluate.py --dataset BloodMNIST --descriptor betti_curves
    python benchmarks/benchmark4/evaluate.py --dataset BloodMNIST --descriptor betti_curves --classifiers TabPFN,CatBoost
    python benchmarks/benchmark4/evaluate.py --dataset BloodMNIST --all-descriptors
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import gc
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, recall_score

from RuleBenchmark.benchmark4.config import (
    DATASETS, ALL_DESCRIPTORS, DESCRIPTORS, ACTIVE_CLASSIFIERS,
    EVALUATION, RAW_RESULTS_PATH, PH_BASED_DESCRIPTORS,
    IMAGE_BASED_DESCRIPTORS, LEARNED_DESCRIPTORS,
)
from RuleBenchmark.benchmark4.optimal_rules import get_rules
from RuleBenchmark.benchmark4.precompute_ph import load_cached_ph, diagrams_to_dict_list
from RuleBenchmark.benchmark4.descriptor_runner import (
    extract_features, extract_features_per_channel, extract_learned_features,
    PH_BASED, IMAGE_BASED, LEARNED,
)
from RuleBenchmark.benchmark4.classifier_wrapper import get_classifier, get_available_classifiers
from RuleBenchmark.benchmark4.data_loader import (
    load_dataset, rgb_to_channels, _to_grayscale_float, prepare_batched_loader,
)


def _build_clf_result(fold_bal_acc, fold_acc, fold_f1, fold_confidence,
                      fold_per_class_recall, t_clf):
    """Build result dict with all metrics for one classifier."""
    return {
        'balanced_accuracy_mean': round(float(np.mean(fold_bal_acc)), 6),
        'balanced_accuracy_std': round(float(np.std(fold_bal_acc)), 6),
        'balanced_accuracy_folds': [round(float(s), 6) for s in fold_bal_acc],
        'accuracy_mean': round(float(np.mean(fold_acc)), 6),
        'accuracy_std': round(float(np.std(fold_acc)), 6),
        'accuracy_folds': [round(float(s), 6) for s in fold_acc],
        'macro_f1_mean': round(float(np.mean(fold_f1)), 6),
        'macro_f1_std': round(float(np.std(fold_f1)), 6),
        'macro_f1_folds': [round(float(s), 6) for s in fold_f1],
        'confidence_mean': round(float(np.mean([c for c in fold_confidence if c >= 0])), 6) if any(c >= 0 for c in fold_confidence) else -1.0,
        'confidence_folds': [round(float(c), 6) for c in fold_confidence],
        'per_class_recall_folds': fold_per_class_recall,
        'time_seconds': round(t_clf, 1),
    }


def _catboost_fold_worker(X_train, y_train, X_test, n_features, n_classes,
                          seed, device, tmp_dir):
    """Worker function for CatBoost GPU in subprocess (isolates C++ crashes)."""
    try:
        clf = get_classifier('CatBoost', n_features=n_features,
                             n_classes=n_classes, seed=seed, device=device)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        np.save(os.path.join(tmp_dir, 'pred.npy'), y_pred)
        try:
            proba = clf.predict_proba(X_test)
            np.save(os.path.join(tmp_dir, 'proba.npy'), proba)
        except Exception:
            pass
    except Exception as e:
        with open(os.path.join(tmp_dir, 'error.txt'), 'w') as f:
            f.write(str(e))


def _run_catboost_fold(X_train, y_train, X_test, n_features, n_classes,
                       seed, device, timeout=600):
    """Run one CatBoost fold in subprocess. Returns (y_pred, y_proba) or raises."""
    tmp_dir = tempfile.mkdtemp(prefix='b4_cb_')
    try:
        ctx = mp.get_context('spawn')
        p = ctx.Process(target=_catboost_fold_worker,
                        args=(X_train, y_train, X_test, n_features, n_classes,
                              seed, device, tmp_dir))
        p.start()
        p.join(timeout=timeout)

        err_path = os.path.join(tmp_dir, 'error.txt')
        pred_path = os.path.join(tmp_dir, 'pred.npy')

        if p.exitcode is None:
            p.kill()
            raise RuntimeError("CatBoost timed out")
        if os.path.exists(err_path):
            with open(err_path) as f:
                raise RuntimeError(f.read())
        if p.exitcode != 0:
            raise RuntimeError(f"CatBoost GPU crashed (exit code {p.exitcode})")
        if not os.path.exists(pred_path):
            raise RuntimeError("CatBoost produced no output")

        y_pred = np.load(pred_path)
        proba_path = os.path.join(tmp_dir, 'proba.npy')
        y_proba = np.load(proba_path) if os.path.exists(proba_path) else None
        return y_pred, y_proba
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def evaluate_one(
    dataset: str,
    descriptor: str,
    classifiers: list = None,
    n_folds: int = 5,
    seed: int = 42,
    device: str = 'cuda',
    output_dir: Path = None,
    n_samples: int = None,
) -> dict:
    """Evaluate one (dataset, descriptor) pair with all classifiers.

    Returns dict with results for each classifier.
    """
    cfg = DATASETS[dataset]
    rules = get_rules()
    object_type = cfg['object_type']
    color_mode = cfg['color_mode']
    n_classes = cfg['n_classes']

    if classifiers is None:
        classifiers = get_available_classifiers()

    if output_dir is None:
        output_dir = RAW_RESULTS_PATH

    # Get optimal params
    desc_cfg = rules.get_descriptor_config(descriptor, object_type, color_mode)
    params = desc_cfg['params']
    dim_per_channel = desc_cfg['dim_per_channel']
    total_dim = desc_cfg['total_dim']

    print(f"\n{'='*70}")
    print(f"  {dataset} x {descriptor}")
    print(f"  object_type={object_type} color={color_mode} dim={total_dim}")
    print(f"  params: {params}")
    print(f"{'='*70}")

    t_start = time.time()

    # ---- Load data ----
    effective_n = n_samples if n_samples is not None else cfg['n_samples']
    if descriptor in PH_BASED:
        ph_data = load_cached_ph(dataset, n_samples=n_samples)
        labels = ph_data['labels']
        diags_gray = ph_data['diagrams_gray_sublevel']

        if color_mode == 'per_channel':
            diags_R = ph_data['diagrams_R_sublevel']
            diags_G = ph_data['diagrams_G_sublevel']
            diags_B = ph_data['diagrams_B_sublevel']
        images = None
    else:
        # Image-based descriptor: use batched loader to avoid OOM on large datasets
        _img_labels, _img_class_names, _load_batch_fn = prepare_batched_loader(
            dataset, n_samples=effective_n, seed=seed)
        labels = _img_labels
        diags_gray = None

    # ---- Determine if learned descriptor (needs per-fold fitting) ----
    is_learned = descriptor in LEARNED

    # ---- Extract features (non-learned descriptors: extract once) ----
    features = None
    if not is_learned:
        print(f"  Extracting features...")
        t_feat = time.time()

        if descriptor in PH_BASED:
            if color_mode == 'per_channel':
                features = extract_features_per_channel(
                    descriptor,
                    diags_R=diags_R, diags_G=diags_G, diags_B=diags_B,
                    params=params, expected_dim_per_channel=dim_per_channel)
            else:
                features = extract_features(
                    descriptor, diagrams=diags_gray,
                    params=params, expected_dim=dim_per_channel)
        else:
            # Image-based: batched extraction to avoid OOM on large datasets
            IMG_BATCH = 500
            actual_n_img = len(labels)
            feat_batches = []
            n_img_batches = (actual_n_img + IMG_BATCH - 1) // IMG_BATCH
            for bi in range(n_img_batches):
                s = bi * IMG_BATCH
                e = min(s + IMG_BATCH, actual_n_img)
                imgs = _load_batch_fn(s, e)
                if color_mode == 'per_channel' and imgs.ndim == 4:
                    R, G, B = rgb_to_channels(imgs)
                    fb = extract_features_per_channel(
                        descriptor,
                        images_R=R, images_G=G, images_B=B,
                        params=params, expected_dim_per_channel=dim_per_channel)
                else:
                    gray = _to_grayscale_float(imgs) if imgs.ndim == 4 else imgs
                    fb = extract_features(
                        descriptor, images=gray,
                        params=params, expected_dim=dim_per_channel)
                feat_batches.append(fb)
                del imgs
                gc.collect()
                if n_img_batches > 1:
                    print(f"      Batch {bi+1}/{n_img_batches}: {fb.shape}")
            features = np.concatenate(feat_batches, axis=0)

        t_feat = time.time() - t_feat
        print(f"    Features: shape={features.shape}, time={t_feat:.1f}s")
        print(f"    NaN count: {np.isnan(features).sum()}, "
              f"range=[{np.nanmin(features):.4f}, {np.nanmax(features):.4f}]")

        # Replace NaN/Inf and clip extreme values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.clip(features, -1e10, 1e10).astype(np.float32)
        # Free source data to reduce memory pressure
        if descriptor in PH_BASED:
            del ph_data
            diags_gray = None
            if color_mode == 'per_channel':
                diags_R = diags_G = diags_B = None
        else:
            _load_batch_fn = None
        gc.collect()
    else:
        print(f"  Learned descriptor: features will be extracted per-fold")

    # ---- Cross-validation ----
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    classifier_results = {}

    # Load existing results to merge with (preserves previously completed classifiers)
    output_dir_p = output_dir if output_dir else RAW_RESULTS_PATH
    existing_path = Path(output_dir_p) / f"{dataset}_{descriptor}.json"
    if existing_path.exists():
        try:
            with open(existing_path) as _ef:
                existing_data = json.load(_ef)
            for _ec, _ev in existing_data.get('classifiers', {}).items():
                if _ec not in classifiers:
                    # Preserve classifiers we're NOT re-running
                    classifier_results[_ec] = _ev
            print(f"  Loaded {len(classifier_results)} existing classifier results from prior run")
        except Exception:
            pass

    # Reorder: CatBoost last (subprocess isolation — if GPU crashes, others are saved)
    if 'CatBoost' in classifiers:
        classifiers = [c for c in classifiers if c != 'CatBoost'] + ['CatBoost']

    # Helper: build result dict for incremental saving
    def _save_incremental():
        total_time = time.time() - t_start
        result = {
            'dataset': dataset, 'descriptor': descriptor,
            'object_type': object_type, 'color_mode': color_mode,
            'n_samples': int(len(labels)), 'n_classes': n_classes,
            'params': {k: _serialize(v) for k, v in params.items()},
            'feature_dim_per_channel': dim_per_channel,
            'total_feature_dim': total_dim,
            'actual_feature_dim': int(features.shape[1]) if features is not None else total_dim,
            'feature_extraction_time_seconds': round(feat_time, 1),
            'n_folds': n_folds, 'seed': seed,
            'classifiers': dict(classifier_results),
            'total_time_seconds': round(total_time, 1),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{dataset}_{descriptor}.json"
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        return result, out_path

    def _clear_gpu():
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def _compute_fold_metrics(y_test, y_pred, y_proba=None):
        """Compute all metrics for one fold."""
        ba = balanced_accuracy_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        pcr = recall_score(y_test, y_pred, average=None, zero_division=0)
        if y_proba is not None:
            conf = float(np.mean(np.max(y_proba, axis=1)))
        else:
            conf = -1.0
        return ba, acc, f1, pcr.tolist(), conf

    def _run_classifier_folds(clf_name, fold_data_iter, n_feat, use_subprocess=False):
        """Run all folds for one classifier, return result dict."""
        print(f"\n  [{clf_name}]")
        t_clf = time.time()
        fold_bal_acc, fold_acc, fold_f1, fold_confidence = [], [], [], []
        fold_per_class_recall = []

        for fold_idx, (X_tr, X_te, y_tr, y_te) in enumerate(fold_data_iter()):
            try:
                if use_subprocess:
                    y_pred, y_proba = _run_catboost_fold(
                        X_tr, y_tr, X_te, n_feat, n_classes,
                        seed + fold_idx, device, timeout=600)
                else:
                    clf = get_classifier(clf_name, n_features=n_feat,
                                         n_classes=n_classes,
                                         seed=seed + fold_idx, device=device)
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)
                    try:
                        y_proba = clf.predict_proba(X_te)
                    except Exception:
                        y_proba = None
                ba, acc, f1, pcr, conf = _compute_fold_metrics(y_te, y_pred, y_proba)
                fold_bal_acc.append(ba)
                fold_acc.append(acc)
                fold_f1.append(f1)
                fold_per_class_recall.append(pcr)
                fold_confidence.append(conf)
                print(f"    Fold {fold_idx+1}: bal_acc={ba:.4f} acc={acc:.4f} f1={f1:.4f}")
            except Exception as e:
                print(f"    Fold {fold_idx+1}: ERROR - {e}")
                fold_bal_acc.append(0.0)
                fold_acc.append(0.0)
                fold_f1.append(0.0)
                fold_confidence.append(-1.0)
                fold_per_class_recall.append([])

        t_clf = time.time() - t_clf
        res = _build_clf_result(fold_bal_acc, fold_acc, fold_f1,
                                fold_confidence, fold_per_class_recall, t_clf)
        print(f"    Mean: bal_acc={np.mean(fold_bal_acc):.4f} acc={np.mean(fold_acc):.4f} "
              f"f1={np.mean(fold_f1):.4f} ({t_clf:.1f}s)")
        return res

    # For non-learned: split precomputed features
    # For learned: extract features inside each fold (fit on train only)
    if is_learned:
        fold_features = []
        print(f"  Extracting per-fold features (fit on train)...")
        t_feat = time.time()
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            if color_mode == 'per_channel':
                channel_trains, channel_tests = [], []
                for ch_name, ch_diags in [('R', diags_R), ('G', diags_G), ('B', diags_B)]:
                    train_d = [ch_diags[i] for i in train_idx]
                    test_d = [ch_diags[i] for i in test_idx]
                    Xtr, Xte = extract_learned_features(
                        descriptor, train_d, test_d,
                        params=params, expected_dim=dim_per_channel)
                    channel_trains.append(Xtr)
                    channel_tests.append(Xte)
                X_train = np.concatenate(channel_trains, axis=1)
                X_test = np.concatenate(channel_tests, axis=1)
            else:
                train_diags = [diags_gray[i] for i in train_idx]
                test_diags = [diags_gray[i] for i in test_idx]
                X_train, X_test = extract_learned_features(
                    descriptor, train_diags, test_diags,
                    params=params, expected_dim=dim_per_channel)
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            X_train = np.clip(X_train, -1e10, 1e10).astype(np.float32)
            X_test = np.clip(X_test, -1e10, 1e10).astype(np.float32)
            y_train, y_test = labels[train_idx], labels[test_idx]
            fold_features.append((X_train, X_test, y_train, y_test))
            print(f"    Fold {fold_idx+1}: train={X_train.shape}, test={X_test.shape}")
        feat_time = time.time() - t_feat
        print(f"    Per-fold extraction time: {feat_time:.1f}s")
        actual_dim = fold_features[0][0].shape[1]
        del ph_data
        diags_gray = None
        if color_mode == 'per_channel':
            diags_R = diags_G = diags_B = None
        gc.collect()

        def _learned_fold_iter():
            for item in fold_features:
                yield item

        for clf_name in classifiers:
            _clear_gpu()
            classifier_results[clf_name] = _run_classifier_folds(
                clf_name, _learned_fold_iter, actual_dim,
                use_subprocess=(clf_name == 'CatBoost'))
            _save_incremental()
    else:
        feat_time = t_feat if 'features' in dir() and features is not None else 0.0

        # Precompute fold splits for reuse across classifiers
        fold_splits = list(skf.split(features, labels))

        def _nonlearned_fold_iter():
            for train_idx, test_idx in fold_splits:
                yield (features[train_idx], features[test_idx],
                       labels[train_idx], labels[test_idx])

        for clf_name in classifiers:
            _clear_gpu()
            n_feat = features.shape[1]
            classifier_results[clf_name] = _run_classifier_folds(
                clf_name, _nonlearned_fold_iter, n_feat,
                use_subprocess=(clf_name == 'CatBoost'))
            _save_incremental()

    # ---- Final save ----
    result, out_path = _save_incremental()
    print(f"\n  Saved: {out_path}")

    return result


def _serialize(v):
    """Make values JSON-serializable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def main():
    parser = argparse.ArgumentParser(description="Benchmark4 Evaluation")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--descriptor', type=str, default=None)
    parser.add_argument('--all-descriptors', action='store_true')
    parser.add_argument('--classifiers', type=str, default=None,
                        help='Comma-separated classifier names')
    parser.add_argument('--n-folds', type=int, default=EVALUATION['n_folds'])
    parser.add_argument('--seed', type=int, default=EVALUATION['seed'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Override n_samples (for testing)')
    args = parser.parse_args()

    if args.classifiers:
        classifiers = args.classifiers.split(',')
    else:
        classifiers = None

    output_dir = Path(args.output_dir) if args.output_dir else RAW_RESULTS_PATH

    if args.all_descriptors:
        descriptors = ALL_DESCRIPTORS
    elif args.descriptor:
        descriptors = [args.descriptor]
    else:
        parser.error("Specify --descriptor or --all-descriptors")

    print("=" * 70)
    print("  BENCHMARK4 EVALUATION")
    print("=" * 70)
    print(f"  Dataset: {args.dataset}")
    print(f"  Descriptors: {descriptors}")
    print(f"  Classifiers: {classifiers or 'all available'}")
    print(f"  Available: {get_available_classifiers()}")

    results = {}
    for desc in descriptors:
        try:
            results[desc] = evaluate_one(
                args.dataset, desc,
                classifiers=classifiers,
                n_folds=args.n_folds,
                seed=args.seed,
                device=args.device,
                output_dir=output_dir,
                n_samples=args.n_samples,
            )
        except Exception as e:
            print(f"\n  ERROR {args.dataset} x {desc}: {e}")
            import traceback
            traceback.print_exc()
            results[desc] = {'error': str(e)}

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for desc, res in results.items():
        if 'error' in res:
            print(f"  {desc:35s}  ERROR: {res['error'][:50]}")
        else:
            best_clf = max(res['classifiers'].items(),
                          key=lambda x: x[1]['balanced_accuracy_mean'])
            print(f"  {desc:35s}  best={best_clf[0]} "
                  f"bal_acc={best_clf[1]['balanced_accuracy_mean']:.4f}")


if __name__ == '__main__':
    main()
