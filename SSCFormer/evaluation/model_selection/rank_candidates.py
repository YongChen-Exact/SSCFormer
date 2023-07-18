#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from SSCFormer.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "SSCFormerPlans"

    overwrite_plans = {
        'SSCFormerTrainerV2_2': ["SSCFormerPlans", "SSCFormerPlansisoPatchesInVoxels"], # r
        'SSCFormerTrainerV2': ["SSCFormerPlansnonCT", "SSCFormerPlansCT2", "SSCFormerPlansallConv3x3",
                            "SSCFormerPlansfixedisoPatchesInVoxels", "SSCFormerPlanstargetSpacingForAnisoAxis",
                            "SSCFormerPlanspoolBasedOnSpacing", "SSCFormerPlansfixedisoPatchesInmm", "SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_warmup': ["SSCFormerPlans", "SSCFormerPlansv2.1", "SSCFormerPlansv2.1_big", "SSCFormerPlansv2.1_verybig"],
        'SSCFormerTrainerV2_cycleAtEnd': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_cycleAtEnd2': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_reduceMomentumDuringTraining': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_graduallyTransitionFromCEToDice': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_independentScalePerAxis': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_Mish': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_Ranger_lr3en4': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_fp32': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_GN': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_momentum098': ["SSCFormerPlans", "SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_momentum09': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_DP': ["SSCFormerPlansv2.1_verybig"],
        'SSCFormerTrainerV2_DDP': ["SSCFormerPlansv2.1_verybig"],
        'SSCFormerTrainerV2_FRN': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_resample33': ["SSCFormerPlansv2.3"],
        'SSCFormerTrainerV2_O2': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_ResencUNet': ["SSCFormerPlans_FabiansResUNet_v2.1"],
        'SSCFormerTrainerV2_DA2': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_allConv3x3': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_ForceBD': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_ForceSD': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_LReLU_slope_2en1': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_lReLU_convReLUIN': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_ReLU': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_ReLU_biasInSegOutput': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_ReLU_convReLUIN': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_lReLU_biasInSegOutput': ["SSCFormerPlansv2.1"],
        #'SSCFormerTrainerV2_Loss_MCC': ["SSCFormerPlansv2.1"],
        #'SSCFormerTrainerV2_Loss_MCCnoBG': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_Loss_DicewithBG': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_Loss_Dice_LR1en3': ["SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_Loss_Dice': ["SSCFormerPlans", "SSCFormerPlansv2.1"],
        'SSCFormerTrainerV2_Loss_DicewithBG_LR1en3': ["SSCFormerPlansv2.1"],
        # 'SSCFormerTrainerV2_fp32': ["SSCFormerPlansv2.1"],
        # 'SSCFormerTrainerV2_fp32': ["SSCFormerPlansv2.1"],
        # 'SSCFormerTrainerV2_fp32': ["SSCFormerPlansv2.1"],
        # 'SSCFormerTrainerV2_fp32': ["SSCFormerPlansv2.1"],
        # 'SSCFormerTrainerV2_fp32': ["SSCFormerPlansv2.1"],

    }

    trainers = ['SSCFormerTrainer'] + ['SSCFormerTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'SSCFormerTrainerNewCandidate24_2',
        'SSCFormerTrainerNewCandidate24_3',
        'SSCFormerTrainerNewCandidate26_2',
        'SSCFormerTrainerNewCandidate27_2',
        'SSCFormerTrainerNewCandidate23_always3DDA',
        'SSCFormerTrainerNewCandidate23_corrInit',
        'SSCFormerTrainerNewCandidate23_noOversampling',
        'SSCFormerTrainerNewCandidate23_softDS',
        'SSCFormerTrainerNewCandidate23_softDS2',
        'SSCFormerTrainerNewCandidate23_softDS3',
        'SSCFormerTrainerNewCandidate23_softDS4',
        'SSCFormerTrainerNewCandidate23_2_fp16',
        'SSCFormerTrainerNewCandidate23_2',
        'SSCFormerTrainerVer2',
        'SSCFormerTrainerV2_2',
        'SSCFormerTrainerV2_3',
        'SSCFormerTrainerV2_3_CE_GDL',
        'SSCFormerTrainerV2_3_dcTopk10',
        'SSCFormerTrainerV2_3_dcTopk20',
        'SSCFormerTrainerV2_3_fp16',
        'SSCFormerTrainerV2_3_softDS4',
        'SSCFormerTrainerV2_3_softDS4_clean',
        'SSCFormerTrainerV2_3_softDS4_clean_improvedDA',
        'SSCFormerTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'SSCFormerTrainerV2_3_softDS4_radam',
        'SSCFormerTrainerV2_3_softDS4_radam_lowerLR',

        'SSCFormerTrainerV2_2_schedule',
        'SSCFormerTrainerV2_2_schedule2',
        'SSCFormerTrainerV2_2_clean',
        'SSCFormerTrainerV2_2_clean_improvedDA_newElDef',

        'SSCFormerTrainerV2_2_fixes', # running
        'SSCFormerTrainerV2_BN', # running
        'SSCFormerTrainerV2_noDeepSupervision', # running
        'SSCFormerTrainerV2_softDeepSupervision', # running
        'SSCFormerTrainerV2_noDataAugmentation', # running
        'SSCFormerTrainerV2_Loss_CE', # running
        'SSCFormerTrainerV2_Loss_CEGDL',
        'SSCFormerTrainerV2_Loss_Dice',
        'SSCFormerTrainerV2_Loss_DiceTopK10',
        'SSCFormerTrainerV2_Loss_TopK10',
        'SSCFormerTrainerV2_Adam', # running
        'SSCFormerTrainerV2_Adam_SSCFormerTrainerlr', # running
        'SSCFormerTrainerV2_SGD_ReduceOnPlateau', # running
        'SSCFormerTrainerV2_SGD_lr1en1', # running
        'SSCFormerTrainerV2_SGD_lr1en3', # running
        'SSCFormerTrainerV2_fixedNonlin', # running
        'SSCFormerTrainerV2_GeLU', # running
        'SSCFormerTrainerV2_3ConvPerStage',
        'SSCFormerTrainerV2_NoNormalization',
        'SSCFormerTrainerV2_Adam_ReduceOnPlateau',
        'SSCFormerTrainerV2_fp16',
        'SSCFormerTrainerV2', # see overwrite_plans
        'SSCFormerTrainerV2_noMirroring',
        'SSCFormerTrainerV2_momentum09',
        'SSCFormerTrainerV2_momentum095',
        'SSCFormerTrainerV2_momentum098',
        'SSCFormerTrainerV2_warmup',
        'SSCFormerTrainerV2_Loss_Dice_LR1en3',
        'SSCFormerTrainerV2_NoNormalization_lr1en3',
        'SSCFormerTrainerV2_Loss_Dice_squared',
        'SSCFormerTrainerV2_newElDef',
        'SSCFormerTrainerV2_fp32',
        'SSCFormerTrainerV2_cycleAtEnd',
        'SSCFormerTrainerV2_reduceMomentumDuringTraining',
        'SSCFormerTrainerV2_graduallyTransitionFromCEToDice',
        'SSCFormerTrainerV2_insaneDA',
        'SSCFormerTrainerV2_independentScalePerAxis',
        'SSCFormerTrainerV2_Mish',
        'SSCFormerTrainerV2_Ranger_lr3en4',
        'SSCFormerTrainerV2_cycleAtEnd2',
        'SSCFormerTrainerV2_GN',
        'SSCFormerTrainerV2_DP',
        'SSCFormerTrainerV2_FRN',
        'SSCFormerTrainerV2_resample33',
        'SSCFormerTrainerV2_O2',
        'SSCFormerTrainerV2_ResencUNet',
        'SSCFormerTrainerV2_DA2',
        'SSCFormerTrainerV2_allConv3x3',
        'SSCFormerTrainerV2_ForceBD',
        'SSCFormerTrainerV2_ForceSD',
        'SSCFormerTrainerV2_ReLU',
        'SSCFormerTrainerV2_LReLU_slope_2en1',
        'SSCFormerTrainerV2_lReLU_convReLUIN',
        'SSCFormerTrainerV2_ReLU_biasInSegOutput',
        'SSCFormerTrainerV2_ReLU_convReLUIN',
        'SSCFormerTrainerV2_lReLU_biasInSegOutput',
        'SSCFormerTrainerV2_Loss_DicewithBG_LR1en3',
        #'SSCFormerTrainerV2_Loss_MCCnoBG',
        'SSCFormerTrainerV2_Loss_DicewithBG',
        # 'SSCFormerTrainerV2_Loss_Dice_LR1en3',
        # 'SSCFormerTrainerV2_Ranger_lr3en4',
        # 'SSCFormerTrainerV2_Ranger_lr3en4',
        # 'SSCFormerTrainerV2_Ranger_lr3en4',
        # 'SSCFormerTrainerV2_Ranger_lr3en4',
        # 'SSCFormerTrainerV2_Ranger_lr3en4',
        # 'SSCFormerTrainerV2_Ranger_lr3en4',
        # 'SSCFormerTrainerV2_Ranger_lr3en4',
        # 'SSCFormerTrainerV2_Ranger_lr3en4',
        # 'SSCFormerTrainerV2_Ranger_lr3en4',
        # 'SSCFormerTrainerV2_Ranger_lr3en4',
        # 'SSCFormerTrainerV2_Ranger_lr3en4',
        # 'SSCFormerTrainerV2_Ranger_lr3en4',
        # 'SSCFormerTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
