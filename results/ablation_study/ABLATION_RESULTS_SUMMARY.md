==================================================================================================================================
  ABLATION STUDY RESULTS — TopoAgent v8.1
  5 conditions × 26 datasets × 200 eval images × 3-fold CV
==================================================================================================================================

1. OVERALL ACCURACY TABLE (Balanced Accuracy %)
----------------------------------------------------------------------------------------------------------------------------------
Dataset                  C0:Full  C1:NoSkl  C2:NoMem  C3:NoRfl  C4:NoAnl    Best  Descriptor (C0)
----------------------------------------------------------------------------------------------------------------------------------
BloodMNIST                 85.1%     82.9%     85.1%     85.1%     85.1%      C0  template_functions
TissueMNIST                24.1%     19.3%     24.1%     24.1%     28.9%      C4  persistence_statistics
PathMNIST                  82.2%     82.2%     82.2%     82.2%     82.2%      C0  persistence_statistics
OCTMNIST                   43.6%     45.8%     43.6%     43.6%     41.6%      C1  lbp_texture
OrganAMNIST                50.5%     34.0%     50.5%     50.5%     50.5%      C0  minkowski_functionals
RetinaMNIST                34.7%     26.8%     34.7%     34.7%     34.7%      C0  lbp_texture
PneumoniaMNIST             84.0%     75.4%     84.0%     84.0%     84.0%      C0  minkowski_functionals
BreastMNIST                70.9%     70.9%     70.9%     70.9%     59.8%      C0  persistence_statistics
DermaMNIST                 15.1%     29.0%     15.1%     29.0%     15.1%      C1  ATOL
OrganCMNIST                46.4%     43.6%     46.4%     46.4%     46.4%      C0  minkowski_functionals
OrganSMNIST                43.7%     31.1%     43.7%     31.1%     43.7%      C0  minkowski_functionals
ISIC2019                   23.6%     22.9%     21.9%     21.9%     21.9%      C0  lbp_texture
Kvasir                     56.7%     56.7%     56.7%     56.7%     56.7%      C0  persistence_landscapes
BrainTumorMRI              80.0%     74.8%     80.0%     69.2%     80.0%      C0  minkowski_functionals
MURA                       28.9%     28.9%     28.9%     18.4%     28.9%      C0  lbp_texture
BreakHis                   30.2%     30.2%     31.7%     31.7%     31.7%      C2  persistence_landscapes
NCT_CRC_HE                 81.8%     81.8%     81.8%     81.8%     81.8%      C0  persistence_statistics
MalariaCell                94.5%     91.0%     94.5%     94.0%     94.5%      C0  minkowski_functionals
IDRiD                      35.2%     35.2%     35.2%     35.2%     35.2%      C0  persistence_landscapes
PCam                       76.0%     76.0%     76.0%     76.0%     77.0%      C4  persistence_statistics
LC25000                    87.0%     94.5%     96.5%     96.5%     87.0%      C2  ATOL
SIPaKMeD                   84.0%     83.5%     78.9%     80.5%     78.9%      C0  template_functions
AML_Cytomorphology         67.4%     44.4%     67.4%     44.4%     44.4%      C0  minkowski_functionals
APTOS2019                  37.2%     35.0%     37.8%     37.8%     37.8%      C2  persistence_landscapes
GasHisSDB                  83.4%     83.4%     83.4%     83.4%     74.4%      C0  persistence_statistics
Chaoyang                   59.8%     59.8%     59.8%     59.5%     59.8%      C0  persistence_landscapes
----------------------------------------------------------------------------------------------------------------------------------
MEAN                       57.9%     55.3%     58.1%     56.5%     56.2%

Condition wins: C0: 19, C1: 2, C2: 3, C3: 0, C4: 2


2. COMPONENT CONTRIBUTION (C0 accuracy − Cx accuracy = drop when removing component)
----------------------------------------------------------------------------------------------------
Component Removed           Avg Drop   C0 Wins   Cx Wins   Tied   Max Drop   Max Gain
----------------------------------------------------------------------------------------------------
C1: w/o Skills                 +2.6%      13/26       3/26    10     +23.0%     -13.9%
C2: w/o Memory                 -0.2%       2/26       3/26    21      +5.1%      -9.5%
C3: w/o Reflect                +1.4%       8/26       4/26    14     +23.0%     -13.9%
C4: w/o Analyze                +1.7%       6/26       4/26    16     +23.0%      -4.8%


3. RESULTS BY OBJECT TYPE
==================================================================================================================================

  CELLS: Discrete Cells (5 datasets)
  ------------------------------------------------------------------------------------------------------------------------------
  Dataset                  C0:Full  C1:NoSkl  C2:NoMem  C3:NoRfl  C4:NoAnl  Descriptor (C0) → Descriptor changes across conditions
  ------------------------------------------------------------------------------------------------------------------------------
  BloodMNIST                 85.1%     82.9%     85.1%     85.1%     85.1%  template_functions  [C1→persistence_sta]
  TissueMNIST                24.1%     19.3%     24.1%     24.1%     28.9%  persistence_statisti  [C1→tropical_coordi, C4→minkowski_funct]
  PCam                       76.0%     76.0%     76.0%     76.0%     77.0%  persistence_statisti  [C4→template_functi]
  MalariaCell                94.5%     91.0%     94.5%     94.0%     94.5%  minkowski_functional  [C1→persistence_sta, C3→template_functi]
  AML_Cytomorphology         67.4%     44.4%     67.4%     44.4%     44.4%  minkowski_functional  [C1→lbp_texture, C3→lbp_texture, C4→lbp_texture]
  ------------------------------------------------------------------------------------------------------------------------------
  GROUP MEAN                 69.4%     62.7%     69.4%     64.7%     66.0%

    C1: w/o Skills           : avg drop = +6.7%, C0 better in 4/5
    C2: w/o Memory           : avg drop = +0.0%, C0 better in 0/5
    C3: w/o Reflect          : avg drop = +4.7%, C0 better in 2/5
    C4: w/o Analyze          : avg drop = +3.4%, C0 better in 1/5

  GLANDS: Glands & Lumens (7 datasets)
  ------------------------------------------------------------------------------------------------------------------------------
  Dataset                  C0:Full  C1:NoSkl  C2:NoMem  C3:NoRfl  C4:NoAnl  Descriptor (C0) → Descriptor changes across conditions
  ------------------------------------------------------------------------------------------------------------------------------
  PathMNIST                  82.2%     82.2%     82.2%     82.2%     82.2%  persistence_statisti
  Kvasir                     56.7%     56.7%     56.7%     56.7%     56.7%  persistence_landscap
  NCT_CRC_HE                 81.8%     81.8%     81.8%     81.8%     81.8%  persistence_statisti
  BreakHis                   30.2%     30.2%     31.7%     31.7%     31.7%  persistence_landscap  [C2→persistence_sta, C3→persistence_sta, C4→persistence_sta]
  LC25000                    87.0%     94.5%     96.5%     96.5%     87.0%  ATOL  [C1→persistence_ent, C2→persistence_sta, C3→persistence_sta]
  Chaoyang                   59.8%     59.8%     59.8%     59.5%     59.8%  persistence_landscap  [C3→persistence_sta]
  GasHisSDB                  83.4%     83.4%     83.4%     83.4%     74.4%  persistence_statisti  [C4→ATOL]
  ------------------------------------------------------------------------------------------------------------------------------
  GROUP MEAN                 68.7%     69.8%     70.3%     70.2%     67.6%

    C1: w/o Skills           : avg drop = -1.1%, C0 better in 0/7
    C2: w/o Memory           : avg drop = -1.6%, C0 better in 0/7
    C3: w/o Reflect          : avg drop = -1.5%, C0 better in 1/7
    C4: w/o Analyze          : avg drop = +1.1%, C0 better in 1/7

  ORGANS: Organ Shape (8 datasets)
  ------------------------------------------------------------------------------------------------------------------------------
  Dataset                  C0:Full  C1:NoSkl  C2:NoMem  C3:NoRfl  C4:NoAnl  Descriptor (C0) → Descriptor changes across conditions
  ------------------------------------------------------------------------------------------------------------------------------
  OrganAMNIST                50.5%     34.0%     50.5%     50.5%     50.5%  minkowski_functional  [C1→tropical_coordi]
  OrganCMNIST                46.4%     43.6%     46.4%     46.4%     46.4%  minkowski_functional  [C1→persistence_sta]
  OrganSMNIST                43.7%     31.1%     43.7%     31.1%     43.7%  minkowski_functional  [C1→persistence_ima, C3→persistence_ima]
  BrainTumorMRI              80.0%     74.8%     80.0%     69.2%     80.0%  minkowski_functional  [C1→persistence_ent, C3→edge_histogram]
  MURA                       28.9%     28.9%     28.9%     18.4%     28.9%  lbp_texture  [C3→minkowski_funct]
  PneumoniaMNIST             84.0%     75.4%     84.0%     84.0%     84.0%  minkowski_functional  [C1→persistence_ent]
  BreastMNIST                70.9%     70.9%     70.9%     70.9%     59.8%  persistence_statisti  [C4→minkowski_funct]
  OCTMNIST                   43.6%     45.8%     43.6%     43.6%     41.6%  lbp_texture  [C1→persistence_ent, C4→minkowski_funct]
  ------------------------------------------------------------------------------------------------------------------------------
  GROUP MEAN                 56.0%     50.6%     56.0%     51.8%     54.4%

    C1: w/o Skills           : avg drop = +5.4%, C0 better in 5/8
    C2: w/o Memory           : avg drop = +0.0%, C0 better in 0/8
    C3: w/o Reflect          : avg drop = +4.2%, C0 better in 3/8
    C4: w/o Analyze          : avg drop = +1.6%, C0 better in 2/8

  SURFACE: Surface Lesions (3 datasets)
  ------------------------------------------------------------------------------------------------------------------------------
  Dataset                  C0:Full  C1:NoSkl  C2:NoMem  C3:NoRfl  C4:NoAnl  Descriptor (C0) → Descriptor changes across conditions
  ------------------------------------------------------------------------------------------------------------------------------
  DermaMNIST                 15.1%     29.0%     15.1%     29.0%     15.1%  ATOL  [C1→persistence_sta, C3→persistence_sta]
  ISIC2019                   23.6%     22.9%     21.9%     21.9%     21.9%  lbp_texture  [C1→persistence_lan, C2→persistence_sta, C3→persistence_sta, C4→persistence_sta]
  SIPaKMeD                   84.0%     83.5%     78.9%     80.5%     78.9%  template_functions  [C1→persistence_sta, C2→lbp_texture, C3→minkowski_funct, C4→lbp_texture]
  ------------------------------------------------------------------------------------------------------------------------------
  GROUP MEAN                 40.9%     45.1%     38.6%     43.8%     38.6%

    C1: w/o Skills           : avg drop = -4.2%, C0 better in 2/3
    C2: w/o Memory           : avg drop = +2.3%, C0 better in 2/3
    C3: w/o Reflect          : avg drop = -2.9%, C0 better in 2/3
    C4: w/o Analyze          : avg drop = +2.3%, C0 better in 2/3

  VESSELS: Vessel Trees (3 datasets)
  ------------------------------------------------------------------------------------------------------------------------------
  Dataset                  C0:Full  C1:NoSkl  C2:NoMem  C3:NoRfl  C4:NoAnl  Descriptor (C0) → Descriptor changes across conditions
  ------------------------------------------------------------------------------------------------------------------------------
  RetinaMNIST                34.7%     26.8%     34.7%     34.7%     34.7%  lbp_texture  [C1→persistence_lan]
  IDRiD                      35.2%     35.2%     35.2%     35.2%     35.2%  persistence_landscap
  APTOS2019                  37.2%     35.0%     37.8%     37.8%     37.8%  persistence_landscap  [C1→persistence_ent, C2→lbp_texture, C3→lbp_texture, C4→lbp_texture]
  ------------------------------------------------------------------------------------------------------------------------------
  GROUP MEAN                 35.7%     32.3%     35.9%     35.9%     35.9%

    C1: w/o Skills           : avg drop = +3.3%, C0 better in 2/3
    C2: w/o Memory           : avg drop = -0.2%, C0 better in 0/3
    C3: w/o Reflect          : avg drop = -0.2%, C0 better in 0/3
    C4: w/o Analyze          : avg drop = -0.2%, C0 better in 0/3


4. DESCRIPTOR SELECTION TABLE
--------------------------------------------------------------------------------------------------------------------------------------------
Dataset                               C0:Full             C1:NoSkill               C2:NoMem              C3:NoRefl              C4:NoAnlz
--------------------------------------------------------------------------------------------------------------------------------------------
BloodMNIST                 template_functions  *persistence_statisti     template_functions     template_functions     template_functions
TissueMNIST              persistence_statisti  *tropical_coordinates   persistence_statisti   persistence_statisti  *minkowski_functional
PathMNIST                persistence_statisti   persistence_statisti   persistence_statisti   persistence_statisti   persistence_statisti
OCTMNIST                          lbp_texture   *persistence_entropy            lbp_texture            lbp_texture  *minkowski_functional
OrganAMNIST              minkowski_functional  *tropical_coordinates   minkowski_functional   minkowski_functional   minkowski_functional
RetinaMNIST                       lbp_texture  *persistence_landscap            lbp_texture            lbp_texture            lbp_texture
PneumoniaMNIST           minkowski_functional   *persistence_entropy   minkowski_functional   minkowski_functional   minkowski_functional
BreastMNIST              persistence_statisti   persistence_statisti   persistence_statisti   persistence_statisti  *minkowski_functional
DermaMNIST                               ATOL  *persistence_statisti                   ATOL  *persistence_statisti                   ATOL
OrganCMNIST              minkowski_functional  *persistence_statisti   minkowski_functional   minkowski_functional   minkowski_functional
OrganSMNIST              minkowski_functional     *persistence_image   minkowski_functional     *persistence_image   minkowski_functional
ISIC2019                          lbp_texture  *persistence_landscap  *persistence_statisti  *persistence_statisti  *persistence_statisti
Kvasir                   persistence_landscap   persistence_landscap   persistence_landscap   persistence_landscap   persistence_landscap
BrainTumorMRI            minkowski_functional   *persistence_entropy   minkowski_functional        *edge_histogram   minkowski_functional
MURA                              lbp_texture            lbp_texture            lbp_texture  *minkowski_functional            lbp_texture
BreakHis                 persistence_landscap   persistence_landscap  *persistence_statisti  *persistence_statisti  *persistence_statisti
NCT_CRC_HE               persistence_statisti   persistence_statisti   persistence_statisti   persistence_statisti   persistence_statisti
MalariaCell              minkowski_functional  *persistence_statisti   minkowski_functional    *template_functions   minkowski_functional
IDRiD                    persistence_landscap   persistence_landscap   persistence_landscap   persistence_landscap   persistence_landscap
PCam                     persistence_statisti   persistence_statisti   persistence_statisti   persistence_statisti    *template_functions
LC25000                                  ATOL   *persistence_entropy  *persistence_statisti  *persistence_statisti                   ATOL
SIPaKMeD                   template_functions  *persistence_statisti           *lbp_texture  *minkowski_functional           *lbp_texture
AML_Cytomorphology       minkowski_functional           *lbp_texture   minkowski_functional           *lbp_texture           *lbp_texture
APTOS2019                persistence_landscap   *persistence_entropy           *lbp_texture           *lbp_texture           *lbp_texture
GasHisSDB                persistence_statisti   persistence_statisti   persistence_statisti   persistence_statisti                  *ATOL
Chaoyang                 persistence_landscap   persistence_landscap   persistence_landscap  *persistence_statisti   persistence_landscap
--------------------------------------------------------------------------------------------------------------------------------------------
* = differs from C0.  Changes: C1: 16/26, C2: 5/26, C3: 12/26, C4: 10/26


5. DESCRIPTOR DIVERSITY
----------------------------------------------------------------------------------------------------
  C0: Full                 : 6 unique — top 3: minkowski_functionals(7), persistence_statistics(6), persistence_landscapes(5)
  C1: w/o Skills           : 6 unique — top 3: persistence_statistics(10), persistence_landscapes(6), persistence_entropy(5)
  C2: w/o Memory           : 6 unique — top 3: persistence_statistics(9), minkowski_functionals(7), lbp_texture(5)
  C3: w/o Reflect          : 7 unique — top 3: persistence_statistics(11), minkowski_functionals(5), lbp_texture(4)
  C4: w/o Analyze          : 6 unique — top 3: minkowski_functionals(9), lbp_texture(5), persistence_statistics(4)


6. CLASSIFIER USAGE
------------------------------------------------------------
  TabPFN         : 110/130 jobs
  XGBoost        : 20/130 jobs
