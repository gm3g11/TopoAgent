# Benchmark4 Results Summary

Generated: 2026-02-10 12:16

## Completion Status
- **Results**: 381 / 390 (97.7%)
- **Missing**: 9
- **Errors**: 0
- **Datasets**: 26
- **Descriptors**: 15
- **Classifiers**: 6 (TabPFN, XGBoost, CatBoost, RandomForest, TabM, RealMLP)

## Classifier Comparison

### Overall (mean balanced accuracy across all evaluations)
| Classifier | Mean | Std | Median | N | #Best |
|------------|------|-----|--------|---|-------|
| TabPFN | 0.6711 | 0.2534 | 0.7354 | 380 | 239 |
| TabM | 0.6491 | 0.2461 | 0.7016 | 380 | 12 |
| XGBoost | 0.6468 | 0.2401 | 0.6923 | 380 | 9 |
| CatBoost | 0.6438 | 0.2429 | 0.6828 | 380 | 113 |
| RealMLP | 0.6405 | 0.2489 | 0.6846 | 380 | 1 |
| RandomForest | 0.6120 | 0.2474 | 0.6494 | 381 | 7 |

### By Object Type (mean balanced accuracy)
| Classifier | discrete_cells | glands_lumens | organ_shape | surface_lesions | vessel_trees |
|------------|---|---|---|---|---|
| TabPFN | 0.7001 | 0.8233 | 0.6988 | 0.5045 | 0.3993 |
| XGBoost | 0.6800 | 0.7697 | 0.6767 | 0.4866 | 0.4120 |
| CatBoost | 0.6793 | 0.7204 | 0.6822 | 0.5259 | 0.4294 |
| RandomForest | 0.6541 | 0.7210 | 0.6469 | 0.4370 | 0.3847 |
| TabM | 0.6847 | 0.7896 | 0.6737 | 0.4965 | 0.3814 |
| RealMLP | 0.6808 | 0.7805 | 0.6632 | 0.4806 | 0.3767 |

## Descriptor Rankings

### Overall (best-classifier accuracy, averaged across datasets)
| Rank | Descriptor | Mean | Std | N |
|------|-----------|------|-----|---|
| 1 | ATOL | 0.7709 | 0.2200 | 22 |
| 2 | persistence_codebook | 0.7553 | 0.2218 | 22 |
| 3 | minkowski_functionals | 0.7514 | 0.2180 | 25 |
| 4 | persistence_statistics | 0.7478 | 0.2158 | 26 |
| 5 | template_functions | 0.7205 | 0.2224 | 26 |
| 6 | lbp_texture | 0.7196 | 0.2178 | 26 |
| 7 | euler_characteristic_curve | 0.7133 | 0.2318 | 26 |
| 8 | persistence_landscapes | 0.7049 | 0.2263 | 26 |
| 9 | betti_curves | 0.7011 | 0.2277 | 26 |
| 10 | persistence_silhouette | 0.6789 | 0.2366 | 26 |
| 11 | persistence_entropy | 0.6664 | 0.2369 | 26 |
| 12 | tropical_coordinates | 0.6638 | 0.2292 | 26 |
| 13 | persistence_image | 0.6245 | 0.2369 | 26 |
| 14 | edge_histogram | 0.5508 | 0.2454 | 26 |
| 15 | euler_characteristic_transform | 0.5106 | 0.2229 | 26 |

### discrete_cells
| Rank | Descriptor | Mean | N |
|------|-----------|------|---|
| 1 | minkowski_functionals | 0.8324 | 5 |
| 2 | ATOL | 0.7664 | 6 |
| 3 | persistence_codebook | 0.7635 | 6 |
| 4 | template_functions | 0.7536 | 6 |
| 5 | persistence_statistics | 0.7522 | 6 |
| 6 | euler_characteristic_curve | 0.7504 | 6 |
| 7 | betti_curves | 0.7459 | 6 |
| 8 | persistence_landscapes | 0.7447 | 6 |
| 9 | persistence_silhouette | 0.7282 | 6 |
| 10 | persistence_entropy | 0.7125 | 6 |
| 11 | lbp_texture | 0.7060 | 6 |
| 12 | tropical_coordinates | 0.7023 | 6 |
| 13 | persistence_image | 0.6949 | 6 |
| 14 | euler_characteristic_transform | 0.5187 | 6 |
| 15 | edge_histogram | 0.4887 | 6 |

### glands_lumens
| Rank | Descriptor | Mean | N |
|------|-----------|------|---|
| 1 | ATOL | 0.9552 | 4 |
| 2 | persistence_codebook | 0.9346 | 4 |
| 3 | persistence_statistics | 0.9002 | 6 |
| 4 | minkowski_functionals | 0.8857 | 6 |
| 5 | euler_characteristic_curve | 0.8778 | 6 |
| 6 | template_functions | 0.8718 | 6 |
| 7 | betti_curves | 0.8655 | 6 |
| 8 | lbp_texture | 0.8606 | 6 |
| 9 | persistence_landscapes | 0.8526 | 6 |
| 10 | persistence_silhouette | 0.8479 | 6 |
| 11 | persistence_entropy | 0.8419 | 6 |
| 12 | tropical_coordinates | 0.8068 | 6 |
| 13 | persistence_image | 0.8053 | 6 |
| 14 | euler_characteristic_transform | 0.5185 | 6 |
| 15 | edge_histogram | 0.5182 | 6 |

### organ_shape
| Rank | Descriptor | Mean | N |
|------|-----------|------|---|
| 1 | ATOL | 0.7792 | 8 |
| 2 | minkowski_functionals | 0.7731 | 8 |
| 3 | persistence_statistics | 0.7621 | 8 |
| 4 | lbp_texture | 0.7611 | 8 |
| 5 | persistence_codebook | 0.7525 | 8 |
| 6 | edge_histogram | 0.7399 | 8 |
| 7 | template_functions | 0.7359 | 8 |
| 8 | persistence_landscapes | 0.7285 | 8 |
| 9 | euler_characteristic_curve | 0.7215 | 8 |
| 10 | betti_curves | 0.6990 | 8 |
| 11 | persistence_silhouette | 0.6687 | 8 |
| 12 | tropical_coordinates | 0.6558 | 8 |
| 13 | persistence_entropy | 0.6490 | 8 |
| 14 | persistence_image | 0.6054 | 8 |
| 15 | euler_characteristic_transform | 0.5911 | 8 |

### surface_lesions
| Rank | Descriptor | Mean | N |
|------|-----------|------|---|
| 1 | ATOL | 0.7491 | 2 |
| 2 | persistence_codebook | 0.7316 | 2 |
| 3 | persistence_statistics | 0.6301 | 3 |
| 4 | minkowski_functionals | 0.6072 | 3 |
| 5 | euler_characteristic_curve | 0.5805 | 3 |
| 6 | template_functions | 0.5659 | 3 |
| 7 | betti_curves | 0.5601 | 3 |
| 8 | lbp_texture | 0.5513 | 3 |
| 9 | tropical_coordinates | 0.5257 | 3 |
| 10 | persistence_silhouette | 0.5219 | 3 |
| 11 | persistence_landscapes | 0.5199 | 3 |
| 12 | persistence_entropy | 0.5040 | 3 |
| 13 | persistence_image | 0.4713 | 3 |
| 14 | euler_characteristic_transform | 0.4080 | 3 |
| 15 | edge_histogram | 0.3671 | 3 |

### vessel_trees
| Rank | Descriptor | Mean | N |
|------|-----------|------|---|
| 1 | lbp_texture | 0.5225 | 3 |
| 2 | persistence_statistics | 0.5134 | 3 |
| 3 | template_functions | 0.4655 | 3 |
| 4 | tropical_coordinates | 0.4600 | 3 |
| 5 | persistence_landscapes | 0.4524 | 3 |
| 6 | minkowski_functionals | 0.4338 | 3 |
| 7 | persistence_entropy | 0.4316 | 3 |
| 8 | betti_curves | 0.4292 | 3 |
| 9 | persistence_silhouette | 0.4266 | 3 |
| 10 | euler_characteristic_curve | 0.4212 | 3 |
| 11 | edge_histogram | 0.4200 | 3 |
| 12 | persistence_codebook | 0.4062 | 2 |
| 13 | ATOL | 0.4044 | 2 |
| 14 | euler_characteristic_transform | 0.3666 | 3 |
| 15 | persistence_image | 0.3261 | 3 |

## Best Results per Dataset
| Dataset | Object Type | Best Descriptor | Best Classifier | Bal. Acc. | N |
|---------|-------------|-----------------|-----------------|-----------|---|
| BloodMNIST | discrete_cells | persistence_codebook | TabPFN | 0.9796 | 15 |
| TissueMNIST | discrete_cells | ATOL | CatBoost | 0.4061 | 15 |
| PathMNIST | glands_lumens | persistence_statistics | TabPFN | 0.9750 | 15 |
| OCTMNIST | organ_shape | template_functions | CatBoost | 0.6673 | 15 |
| OrganAMNIST | organ_shape | minkowski_functionals | TabPFN | 0.8558 | 15 |
| RetinaMNIST | vessel_trees | persistence_statistics | TabPFN | 0.5553 | 15 |
| PneumoniaMNIST | organ_shape | edge_histogram | CatBoost | 0.9540 | 15 |
| BreastMNIST | organ_shape | persistence_statistics | CatBoost | 0.7898 | 15 |
| DermaMNIST | surface_lesions | persistence_statistics | TabPFN | 0.5765 | 15 |
| OrganCMNIST | organ_shape | edge_histogram | TabM | 0.8292 | 15 |
| OrganSMNIST | organ_shape | minkowski_functionals | TabPFN | 0.8119 | 15 |
| ISIC2019 | surface_lesions | minkowski_functionals | CatBoost | 0.3686 | 13 |
| Kvasir | glands_lumens | persistence_statistics | TabPFN | 0.7953 | 13 |
| BrainTumorMRI | organ_shape | ATOL | TabPFN | 0.9608 | 15 |
| MURA | organ_shape | lbp_texture | TabPFN | 0.5797 | 15 |
| BreakHis | glands_lumens | persistence_statistics | TabPFN | 0.8871 | 14 |
| NCT_CRC_HE | glands_lumens | persistence_codebook | TabPFN | 0.9771 | 15 |
| MalariaCell | discrete_cells | persistence_codebook | TabPFN | 0.9544 | 15 |
| IDRiD | vessel_trees | lbp_texture | TabPFN | 0.4597 | 15 |
| PCam | discrete_cells | ATOL | TabPFN | 0.9044 | 15 |
| LC25000 | glands_lumens | persistence_codebook | TabPFN | 1.0000 | 15 |
| SIPaKMeD | discrete_cells | persistence_statistics | TabPFN | 0.9764 | 15 |
| AML_Cytomorphology | discrete_cells | betti_curves | TabM | 0.4070 | 14 |
| APTOS2019 | vessel_trees | lbp_texture | TabPFN | 0.5772 | 13 |
| GasHisSDB | surface_lesions | ATOL | TabPFN | 0.9587 | 15 |
| Chaoyang | glands_lumens | persistence_codebook | TabPFN | 0.7883 | 14 |

## Object Type Validation

### discrete_cells (6 datasets: BloodMNIST, TissueMNIST, MalariaCell, PCam, SIPaKMeD, AML_Cytomorphology)
Top 3 descriptors: minkowski_functionals, ATOL, persistence_codebook

### glands_lumens (6 datasets: PathMNIST, Kvasir, BreakHis, NCT_CRC_HE, LC25000, Chaoyang)
Top 3 descriptors: ATOL, persistence_codebook, persistence_statistics

### organ_shape (8 datasets: OCTMNIST, OrganAMNIST, PneumoniaMNIST, BreastMNIST, OrganCMNIST, OrganSMNIST, BrainTumorMRI, MURA)
Top 3 descriptors: ATOL, minkowski_functionals, persistence_statistics

### surface_lesions (3 datasets: DermaMNIST, ISIC2019, GasHisSDB)
Top 3 descriptors: ATOL, persistence_codebook, persistence_statistics

### vessel_trees (3 datasets: RetinaMNIST, IDRiD, APTOS2019)
Top 3 descriptors: lbp_texture, persistence_statistics, template_functions