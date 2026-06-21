#!/usr/bin/env python
"""
Case Study: TopoAgent vs GPT-4o Descriptor Selection

Generates a per-sample comparison showing that TopoAgent's adaptive
descriptor selection leads to correct classification on specific images
where GPT-4o's default descriptor choice fails.

Usage:
    python scripts/case_study_comparison.py --dataset PathMNIST
    python scripts/case_study_comparison.py --dataset BloodMNIST
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Dataset config: TopoAgent choice vs GPT-4o choice ---
DATASET_CONFIG = {
    "BloodMNIST": {
        "topoagent_desc": "template_functions",
        "gpt4o_desc": "persistence_silhouette",
        "object_type": "discrete_cells",
        "class_names": [
            "basophil", "eosinophil", "erythroblast", "immature_granulocyte",
            "lymphocyte", "monocyte", "neutrophil", "platelet"
        ],
        "topoagent_params": {"n_templates": 81, "template_type": "tent"},
        "gpt4o_params": {"resolution": 50},
    },
    "PathMNIST": {
        "topoagent_desc": "persistence_statistics",
        "gpt4o_desc": "persistence_silhouette",
        "object_type": "glands_lumens",
        "class_names": [
            "adipose", "background", "debris", "lymphocytes", "mucus",
            "smooth_muscle", "normal_colon_mucosa", "cancer_stroma", "tumor_epithelium"
        ],
        "topoagent_params": {},
        "gpt4o_params": {"resolution": 50},
    },
    "OrganAMNIST": {
        "topoagent_desc": "minkowski_functionals",
        "gpt4o_desc": "persistence_silhouette",
        "object_type": "organ_shape",
        "class_names": [
            "bladder", "femur_left", "femur_right", "heart", "kidney_left",
            "kidney_right", "liver", "lung_left", "lung_right", "spleen", "pancreas"
        ],
        "topoagent_params": {},
        "gpt4o_params": {"resolution": 50},
    },
    "BreakHis": {
        "topoagent_desc": "persistence_statistics",
        "gpt4o_desc": "persistence_silhouette",
        "object_type": "glands_lumens",
        "class_names": [
            "adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma",
            "ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"
        ],
        "topoagent_params": {},
        "gpt4o_params": {"resolution": 50},
    },
}


def load_dataset(name, n_samples=300):
    """Load dataset via RuleBenchmark/benchmark4 loader or medmnist."""
    # Try RuleBenchmark loader first (handles all 26 datasets)
    try:
        rb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RuleBenchmark', 'benchmark4')
        sys.path.insert(0, rb_path)
        from data_loader import load_dataset as b4_load
        images, labels, class_names_loaded = b4_load(name, n_samples=n_samples)
        return images, labels
    except Exception as e:
        print(f"  RuleBenchmark loader failed: {e}")

    # Fallback: direct medmnist at 224x224
    import medmnist
    from medmnist import INFO
    info = INFO[name.lower()]
    DataClass = getattr(medmnist, info["python_class"])
    dataset = DataClass(split="test", download=True, size=224)
    images = []
    labels = []
    for i in range(min(n_samples, len(dataset))):
        img, lbl = dataset[i]
        img_np = np.array(img).astype(np.float32) / 255.0
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        images.append(img_np)
        labels.append(int(lbl.flatten()[0]))
    return np.array(images), np.array(labels)


def compute_ph_for_image(image):
    """Compute persistent homology using GUDHI cubical complex.

    Returns dict in project format: {"H0": [{"birth": ..., "death": ..., "persistence": ...}, ...], "H1": [...]}
    """
    import gudhi
    from skimage.color import rgb2gray
    from skimage.transform import resize

    if image.ndim == 3:
        gray = rgb2gray(image)
    else:
        gray = image.astype(np.float64)

    if gray.max() > 1.0:
        gray = gray / 255.0

    # Resize large images to 224x224 (preserves glandular detail)
    if gray.shape[0] > 224:
        gray = resize(gray, (224, 224), anti_aliasing=True)

    cc = gudhi.CubicalComplex(
        dimensions=list(gray.shape),
        top_dimensional_cells=gray.flatten()
    )
    cc.compute_persistence()

    persistence_by_dim = {}
    for dim in [0, 1]:
        intervals = cc.persistence_intervals_in_dimension(dim)
        pairs = []
        for birth, death in intervals:
            if death == float('inf'):
                death = float(gray.max())
            if death > birth:
                pairs.append({
                    "birth": float(birth),
                    "death": float(death),
                    "persistence": float(death - birth)
                })
        pairs.sort(key=lambda x: -x["persistence"])
        persistence_by_dim[f"H{dim}"] = pairs

    return persistence_by_dim


def extract_features_batch(images, descriptor_name, params=None, color_mode="per_channel"):
    """Extract features for a batch of images using given descriptor."""
    from topoagent.tools.descriptors import get_all_descriptors

    descs = get_all_descriptors()
    desc_tool = descs[descriptor_name]

    features = []
    valid_indices = []

    # Image-based descriptors that don't use PH data
    IMAGE_BASED_DESCS = {"minkowski_functionals", "euler_characteristic_curve",
                         "euler_characteristic_transform", "lbp_texture", "edge_histogram"}

    def _extract_fv(result):
        """Extract feature vector from tool result."""
        if isinstance(result, dict):
            for key in ["combined_vector", "feature_vector", "features", "vector"]:
                if key in result and result[key] is not None:
                    arr = np.array(result[key]).flatten()
                    if len(arr) > 0:
                        return arr
            # Fallback: find largest array value
            for k, v in result.items():
                if isinstance(v, (list, np.ndarray)) and k not in ("success", "error", "tool_name"):
                    arr = np.array(v).flatten()
                    if len(arr) > 1:
                        return arr
        return None

    def _invoke_desc(desc_tool, diags, params, descriptor_name, image=None):
        """Invoke descriptor with correct arg name."""
        if descriptor_name in IMAGE_BASED_DESCS:
            # Image-based descriptor
            if image is None:
                return None
            img_input = image.astype(np.float64)
            if img_input.max() > 1.0:
                img_input = img_input / 255.0
            try:
                return desc_tool._run(image_array=img_input)
            except Exception:
                return desc_tool._run(image=img_input)
        # PH-based descriptor
        try:
            return desc_tool._run(persistence_data=diags)
        except Exception:
            return desc_tool._run(persistence_diagrams=diags)

    for i, img in enumerate(images):
        try:
            if descriptor_name in IMAGE_BASED_DESCS:
                # Image-based: no PH needed, but may do per-channel
                if color_mode == "per_channel" and img.ndim == 3 and img.shape[-1] == 3:
                    channel_feats = []
                    for ch in range(3):
                        channel_img = img[:, :, ch]
                        result = _invoke_desc(desc_tool, None, params, descriptor_name, image=channel_img)
                        fv = _extract_fv(result)
                        if fv is not None:
                            channel_feats.append(fv)
                    if channel_feats:
                        features.append(np.concatenate(channel_feats))
                        valid_indices.append(i)
                else:
                    gray = img
                    if img.ndim == 3:
                        from skimage.color import rgb2gray
                        gray = rgb2gray(img)
                    result = _invoke_desc(desc_tool, None, params, descriptor_name, image=gray)
                    fv = _extract_fv(result)
                    if fv is not None:
                        features.append(fv)
                        valid_indices.append(i)
            elif color_mode == "per_channel" and img.ndim == 3 and img.shape[-1] == 3:
                channel_feats = []
                for ch in range(3):
                    channel_img = img[:, :, ch]
                    diags = compute_ph_for_image(channel_img)
                    result = _invoke_desc(desc_tool, diags, params, descriptor_name)
                    fv = _extract_fv(result)
                    if fv is not None:
                        channel_feats.append(fv)
                if channel_feats:
                    feat = np.concatenate(channel_feats)
                    features.append(feat)
                    valid_indices.append(i)
            else:
                diags = compute_ph_for_image(img)
                result = _invoke_desc(desc_tool, diags, params, descriptor_name)
                fv = _extract_fv(result)
                if fv is not None:
                    features.append(fv)
                    valid_indices.append(i)
        except Exception as e:
            if i < 3:
                print(f"  [warn] sample {i}: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  Extracted {i + 1}/{len(images)} ({descriptor_name})")

    # Pad/truncate to uniform length
    if not features:
        return np.array([]), valid_indices

    max_len = max(len(f) for f in features)
    uniform = np.zeros((len(features), max_len))
    for j, f in enumerate(features):
        uniform[j, :len(f)] = f

    return uniform, valid_indices


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Classify using TabPFN (no training needed)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Replace NaN/Inf
    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

    # TabPFN: max ~2000 features, max ~10000 train samples, max 10 classes
    n_classes = len(np.unique(y_train))
    feat_dim = X_train_s.shape[1]

    # Truncate features if >2000D for TabPFN
    if feat_dim > 2000:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=500)
        X_train_s = pca.fit_transform(X_train_s)
        X_test_s = pca.transform(X_test_s)

    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(device='cuda', n_estimators=16)
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    proba = clf.predict_proba(X_test_s)
    acc = balanced_accuracy_score(y_test, preds)

    return preds, proba, acc


def find_showcase_samples(topo_preds, gpt4o_preds, labels, images, config, n_show=5):
    """Find samples where TopoAgent is correct but GPT-4o is wrong."""
    showcase = []
    for i in range(len(labels)):
        topo_correct = (topo_preds[i] == labels[i])
        gpt4o_correct = (gpt4o_preds[i] == labels[i])
        if topo_correct and not gpt4o_correct:
            showcase.append({
                "index": i,
                "true_label": int(labels[i]),
                "true_class": config["class_names"][labels[i]],
                "topoagent_pred": int(topo_preds[i]),
                "topoagent_class": config["class_names"][topo_preds[i]],
                "gpt4o_pred": int(gpt4o_preds[i]),
                "gpt4o_class": config["class_names"][gpt4o_preds[i]],
            })
    # Sort by how "wrong" GPT-4o was (label distance)
    showcase.sort(key=lambda x: abs(x["true_label"] - x["gpt4o_pred"]), reverse=True)
    return showcase[:n_show]


def main():
    parser = argparse.ArgumentParser(description="Case study comparison")
    parser.add_argument("--dataset", default="PathMNIST",
                       choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--n-train", type=int, default=200,
                       help="Training samples")
    parser.add_argument("--n-test", type=int, default=100,
                       help="Test samples (to find showcase)")
    parser.add_argument("--color-mode", default="per_channel",
                       choices=["per_channel", "grayscale"])
    args = parser.parse_args()

    config = DATASET_CONFIG[args.dataset]
    print(f"=" * 60)
    print(f"CASE STUDY: {args.dataset}")
    print(f"  TopoAgent descriptor: {config['topoagent_desc']}")
    print(f"  GPT-4o descriptor:    {config['gpt4o_desc']}")
    print(f"=" * 60)

    # Load dataset
    print(f"\n[1] Loading {args.dataset}...")
    total = args.n_train + args.n_test
    images, labels = load_dataset(args.dataset, n_samples=total)
    print(f"  Loaded {len(images)} images, {len(np.unique(labels))} classes")
    print(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Split
    X_train_imgs = images[:args.n_train]
    y_train = labels[:args.n_train]
    X_test_imgs = images[args.n_train:args.n_train + args.n_test]
    y_test = labels[args.n_train:args.n_train + args.n_test]
    print(f"  Train: {len(X_train_imgs)}, Test: {len(X_test_imgs)}")

    # Extract features with TopoAgent's descriptor
    print(f"\n[2] Extracting features with TopoAgent descriptor: {config['topoagent_desc']}")
    t0 = time.time()
    topo_train, topo_train_idx = extract_features_batch(
        X_train_imgs, config["topoagent_desc"], config["topoagent_params"], args.color_mode
    )
    topo_test, topo_test_idx = extract_features_batch(
        X_test_imgs, config["topoagent_desc"], config["topoagent_params"], args.color_mode
    )
    t_topo = time.time() - t0
    print(f"  Feature dim: {topo_train.shape[1] if len(topo_train) else 'N/A'}")
    print(f"  Valid train/test: {len(topo_train_idx)}/{len(topo_test_idx)}")
    print(f"  Time: {t_topo:.1f}s")

    # Extract features with GPT-4o's descriptor
    print(f"\n[3] Extracting features with GPT-4o descriptor: {config['gpt4o_desc']}")
    t0 = time.time()
    gpt4o_train, gpt4o_train_idx = extract_features_batch(
        X_train_imgs, config["gpt4o_desc"], config["gpt4o_params"], args.color_mode
    )
    gpt4o_test, gpt4o_test_idx = extract_features_batch(
        X_test_imgs, config["gpt4o_desc"], config["gpt4o_params"], args.color_mode
    )
    t_gpt4o = time.time() - t0
    print(f"  Feature dim: {gpt4o_train.shape[1] if len(gpt4o_train) else 'N/A'}")
    print(f"  Valid train/test: {len(gpt4o_train_idx)}/{len(gpt4o_test_idx)}")
    print(f"  Time: {t_gpt4o:.1f}s")

    # Find common indices
    common_train = sorted(set(topo_train_idx) & set(gpt4o_train_idx))
    common_test = sorted(set(topo_test_idx) & set(gpt4o_test_idx))
    print(f"\n  Common samples: train={len(common_train)}, test={len(common_test)}")

    # Build aligned matrices
    topo_train_map = {idx: i for i, idx in enumerate(topo_train_idx)}
    topo_test_map = {idx: i for i, idx in enumerate(topo_test_idx)}
    gpt4o_train_map = {idx: i for i, idx in enumerate(gpt4o_train_idx)}
    gpt4o_test_map = {idx: i for i, idx in enumerate(gpt4o_test_idx)}

    topo_X_train = np.array([topo_train[topo_train_map[i]] for i in common_train])
    topo_X_test = np.array([topo_test[topo_test_map[i]] for i in common_test])
    gpt4o_X_train = np.array([gpt4o_train[gpt4o_train_map[i]] for i in common_train])
    gpt4o_X_test = np.array([gpt4o_test[gpt4o_test_map[i]] for i in common_test])
    y_tr = np.array([y_train[i] for i in common_train])
    y_te = np.array([y_test[i] for i in common_test])

    # Train and evaluate
    print(f"\n[4] Training classifiers...")
    topo_preds, topo_proba, topo_acc = train_and_evaluate(topo_X_train, y_tr, topo_X_test, y_te)
    gpt4o_preds, gpt4o_proba, gpt4o_acc = train_and_evaluate(gpt4o_X_train, y_tr, gpt4o_X_test, y_te)

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  TopoAgent ({config['topoagent_desc']}): {topo_acc*100:.1f}% balanced accuracy")
    print(f"  GPT-4o    ({config['gpt4o_desc']}):     {gpt4o_acc*100:.1f}% balanced accuracy")
    print(f"  Gap: {(topo_acc - gpt4o_acc)*100:+.1f}%")

    # Per-sample comparison
    topo_correct = (topo_preds == y_te)
    gpt4o_correct = (gpt4o_preds == y_te)
    both_correct = np.sum(topo_correct & gpt4o_correct)
    only_topo = np.sum(topo_correct & ~gpt4o_correct)
    only_gpt4o = np.sum(~topo_correct & gpt4o_correct)
    both_wrong = np.sum(~topo_correct & ~gpt4o_correct)

    print(f"\n  Per-sample breakdown ({len(y_te)} test samples):")
    print(f"    Both correct:      {both_correct}")
    print(f"    Only TopoAgent:    {only_topo}  <-- showcase candidates")
    print(f"    Only GPT-4o:       {only_gpt4o}")
    print(f"    Both wrong:        {both_wrong}")

    # Find showcase samples
    test_images = np.array([X_test_imgs[i] for i in common_test])
    showcase = find_showcase_samples(topo_preds, gpt4o_preds, y_te, test_images, config, n_show=10)

    if showcase:
        print(f"\n{'=' * 60}")
        print(f"SHOWCASE SAMPLES (TopoAgent correct, GPT-4o wrong)")
        print(f"{'=' * 60}")
        for s in showcase:
            print(f"  Sample {s['index']:3d}: "
                  f"True={s['true_class']:25s} | "
                  f"TopoAgent={s['topoagent_class']:25s} ✓ | "
                  f"GPT4o={s['gpt4o_class']:25s} ✗")
    else:
        print("\n  No showcase samples found (both methods agree on all samples)")

    # Save results
    output = {
        "dataset": args.dataset,
        "topoagent_descriptor": config["topoagent_desc"],
        "gpt4o_descriptor": config["gpt4o_desc"],
        "topoagent_accuracy": float(topo_acc),
        "gpt4o_accuracy": float(gpt4o_acc),
        "accuracy_gap": float(topo_acc - gpt4o_acc),
        "n_test": len(y_te),
        "both_correct": int(both_correct),
        "only_topoagent_correct": int(only_topo),
        "only_gpt4o_correct": int(only_gpt4o),
        "both_wrong": int(both_wrong),
        "showcase_samples": showcase,
        "feature_stats": {
            "topoagent_dim": int(topo_X_test.shape[1]),
            "topoagent_variance": float(np.var(topo_X_test)),
            "topoagent_sparsity": float(np.mean(np.abs(topo_X_test) < 1e-6) * 100),
            "gpt4o_dim": int(gpt4o_X_test.shape[1]),
            "gpt4o_variance": float(np.var(gpt4o_X_test)),
            "gpt4o_sparsity": float(np.mean(np.abs(gpt4o_X_test) < 1e-6) * 100),
        }
    }

    out_path = f"results/case_studies/{args.dataset}_descriptor_comparison.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Save showcase images (224x224)
    if showcase:
        img_dir = "results/case_studies/images"
        os.makedirs(img_dir, exist_ok=True)
        from PIL import Image
        for s in showcase[:5]:
            idx = s["index"]
            img = test_images[idx]
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            # Ensure at least 224x224
            if pil_img.size[0] < 224:
                pil_img = pil_img.resize((224, 224), Image.LANCZOS)
            fname = f"{img_dir}/{args.dataset}_showcase_{idx}_true_{s['true_class']}.png"
            pil_img.save(fname)
            print(f"  Saved: {fname} ({pil_img.size[0]}x{pil_img.size[1]})")


if __name__ == "__main__":
    main()
