# TopoAgent Workflow Example

This document shows how TopoAgent thinks, reasons, acts, and reflects during a dermoscopy image classification task.

## Task
**Query**: "Classify this dermoscopy skin lesion image"
**Image**: `temp_eval/sample_393.png`

---

## Round 1: Load Image

### 🤔 THINKING (Tool Selection)
```
LLM analyzes the task and available tools:
- Query requires image classification using TDA
- First step: Load and preprocess the image
- Available tools: image_loader, compute_ph, persistence_image, pytorch_classifier
```

**Decision**: Select `image_loader`

### 🔧 ACTING (Tool Execution)
```python
image_loader(
    image_path="temp_eval/sample_393.png",
    normalize=True,
    grayscale=True
)
```

**Output**:
```json
{
  "success": true,
  "image_array": [[0.88, 0.89, 0.88, ...], ...],  // 28x28 normalized grayscale
  "shape": [28, 28],
  "dtype": "float32"
}
```

### 💭 REFLECTION
```
Error Analysis:
- Tool executed successfully
- Image loaded and preprocessed
- Missing: Topological features not yet computed

Suggestion:
- AUGMENT: "Need to compute persistent homology for topological features"
- Next step should use compute_ph tool
```

### ✅ Completion Check
```json
{
  "is_complete": false,
  "reasoning": "We have loaded the image but not computed persistent
               homology or extracted topological features for classification"
}
```

---

## Round 2: Compute Persistent Homology

### 🤔 THINKING (Tool Selection)
```
LLM reviews short-term memory:
- image_loader completed ✓
- Next: Need topological features

Available data: 28x28 grayscale image array
```

**Decision**: Select `compute_ph`

### 🔧 ACTING (Tool Execution)
```python
compute_ph(
    image_array=[[0.88, 0.89, ...], ...],
    filtration_type="sublevel",
    max_dimension=1
)
```

**Output**:
```json
{
  "success": true,
  "h0_diagram": [[0.0, 0.12], [0.0, 0.15], [0.0, 0.23], ...],  // 45 features
  "h1_diagram": [[0.34, 0.67], [0.41, 0.58], ...],              // 12 features
  "filtration_type": "sublevel"
}
```

### 💭 REFLECTION
```
Error Analysis:
- Persistent homology computed successfully
- H0 (connected components): 45 birth-death pairs
- H1 (loops/holes): 12 birth-death pairs

Suggestion:
- AUGMENT: "Need to vectorize persistence diagrams for classifier input"
- H1 features indicate circular structures in the lesion
```

### ✅ Completion Check
```json
{
  "is_complete": false,
  "reasoning": "Have persistence diagrams but need to convert to
               feature vector for classification"
}
```

---

## Round 3: Extract Topological Features

### 🤔 THINKING (Tool Selection)
```
LLM reviews progress:
- Image loaded ✓
- PH computed ✓
- Need: Feature vectorization

The persistence diagrams need to be converted to a fixed-size
feature vector for the classifier.
```

**Decision**: Select `persistence_image` (or `topological_features`)

### 🔧 ACTING (Tool Execution)
```python
persistence_image(
    h0_diagram=[[0.0, 0.12], ...],
    h1_diagram=[[0.34, 0.67], ...],
    resolution=20,
    sigma=0.1
)
```

**Output**:
```json
{
  "success": true,
  "feature_vector": [0.023, 0.045, 0.012, ...],  // 800 dimensions
  "h0_image_shape": [20, 20],
  "h1_image_shape": [20, 20]
}
```

### 💭 REFLECTION
```
Error Analysis:
- Feature extraction successful
- 800D feature vector ready for classification

Suggestion:
- PROCEED: "All preprocessing complete, ready for classification"
- Features capture: component distribution (H0) + loop structures (H1)
```

### ✅ Completion Check
```json
{
  "is_complete": false,
  "reasoning": "Features extracted, but final classification not yet made"
}
```

---

## Round 4: Classify with Trained Model

### 🤔 THINKING (Tool Selection)
```
LLM reviews pipeline:
- Image loaded ✓
- PH computed ✓
- Features extracted ✓
- Remaining: Classification

Feature vector (800D) is ready for the PyTorch classifier.
```

**Decision**: Select `pytorch_classifier`

### 🔧 ACTING (Tool Execution)
```python
pytorch_classifier(
    feature_vector=[0.023, 0.045, 0.012, ...],
    model_path="models/dermamnist_pi_mlp.pt"
)
```

**Output**:
```json
{
  "success": true,
  "predicted_class": 5,
  "predicted_label": "melanocytic nevi",
  "confidence": 0.73,
  "class_probabilities": {
    "0": 0.02,  // actinic keratoses
    "1": 0.05,  // basal cell carcinoma
    "2": 0.08,  // benign keratosis
    "3": 0.01,  // dermatofibroma
    "4": 0.11,  // melanoma
    "5": 0.73,  // melanocytic nevi ← PREDICTED
    "6": 0.00   // vascular lesions
  }
}
```

### 💭 REFLECTION
```
Error Analysis:
- Classification successful
- High confidence (73%) for melanocytic nevi
- Second highest: melanoma (11%) - important differential

Suggestion:
- COMPLETE: Classification task finished with confident prediction
```

### ✅ Completion Check
```json
{
  "is_complete": true,
  "reasoning": "Successfully classified the image using TDA pipeline:
               1. Image loaded and preprocessed
               2. Persistent homology computed (H0: 45, H1: 12 features)
               3. Persistence image feature vector extracted (800D)
               4. PyTorch classifier predicted: Class 5 (melanocytic nevi)
               with 73% confidence"
}
```

---

## Final Answer Generation

```
Classification: melanocytic nevi (Class 5)
Confidence: 73%

Topological Evidence:
- H0 features (45 components): Indicates uniform pigment distribution
- H1 features (12 loops): Consistent with round/oval shape, clear borders

Reasoning Trace:
1. Image loaded and converted to grayscale
2. Computed persistent homology using sublevel filtration
3. Generated 800D persistence image feature vector
4. PyTorch MLP classifier predicted melanocytic nevi

Tools Used: image_loader → compute_ph → persistence_image → pytorch_classifier
```

---

## Memory State at End

### Short-Term Memory (Session-Scoped)
```
[
  ("image_loader", {success: true, shape: [28,28]}),
  ("compute_ph", {success: true, h0: 45, h1: 12}),
  ("persistence_image", {success: true, dim: 800}),
  ("pytorch_classifier", {success: true, class: 5, conf: 0.73})
]
```

### Long-Term Memory (Reflections)
```
Round 1: "Image loaded, need to compute PH next"
Round 2: "PH computed, H1 features indicate circular structures"
Round 3: "Features extracted, ready for classification"
Round 4: "Classification complete with high confidence"
```

---

## Why This Approach Works

1. **Structured Reasoning**: LLM breaks down the task into logical steps
2. **Tool Orchestration**: Automatically selects appropriate tools based on current state
3. **Reflection Loop**: Evaluates progress and adjusts strategy after each step
4. **Memory Tracking**: Maintains context across rounds for informed decisions
5. **Trained Classifier**: Final prediction uses a properly trained model, not LLM heuristics

The key insight: **TDA features capture structural patterns** (borders, internal structures) that pure vision models miss, while the **reflection mechanism** ensures the pipeline executes correctly.
