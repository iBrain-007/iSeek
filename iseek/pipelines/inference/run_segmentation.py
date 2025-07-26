import os
from typing import Tuple, Union
from rich import print as rprint
import prettytable
import pandas as pd
from PIL import Image
import torch
from torch._dynamo import OptimizedModule
import iseek
from batchgenerators.utilities.file_and_folder_operations import load_json, join, maybe_mkdir_p, isdir
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from iseek.pipelines.evaluation.evaluation_vessel_seg import performance_evaluation, color_predictions


class iSeekSegmentor(nnUNetPredictor):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        super().__init__(tile_step_size, use_gaussian, use_mirroring, perform_everything_on_device,
                         device, verbose, verbose_preprocessing, allow_tqdm)

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(iseek.__path__[0], "pipelines", "training", "trainers"),
                                                    trainer_name, 'iseek.pipelines.training.trainers')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in iseek.pipelines.training.trainers. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        network.load_state_dict(parameters[0])
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)


def evaluation_after_segmentation(pred_dir: str,
                                  target_dir: str,
                                  exclude_background: bool,
                                  num_classes: int,
                                  workers: int = os.cpu_count()):
    pred_dir = os.path.abspath(pred_dir)
    target_dir = os.path.abspath(target_dir)

    df = performance_evaluation(pred_dir, target_dir,
                                exclude_background=exclude_background,
                                num_classes=num_classes,
                                workers=workers)

    out_xlsx = os.path.join(os.path.dirname(pred_dir), 'metrics_for_each.xlsx')
    df.to_excel(out_xlsx, index=False)
    rprint(f"[green]Per‑image metrics saved to:[/green] {out_xlsx}")

    summary = {k: (df[k].mean(), df[k].std()) for k in df.columns}
    tbl = prettytable.PrettyTable()
    tbl.field_names = df.columns
    tbl.add_row([f"{m:.3f}±{s:.3f}" for m, s in summary.values()])

    txt_path = os.path.join(os.path.dirname(pred_dir), 'overall_performance.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(tbl.get_string())

    rprint(tbl)
    rprint("[green]Evaluation complete. Generating visualizations…[/green]")
    color_predictions(pred_dir)
    rprint("[green]Visualization complete.[/green]")


def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='iSeekUltraS3',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--eval', action='store_true', help="Set this if you want to evaluate after prediction.", )
    parser.add_argument('-gt', type=str, required=False, default=None,
                        help="Ground truth directory if you want to evaluate")
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_best.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_best.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)
    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'
    assert args.device in ['cpu', 'cuda', 'mps'], (f'-device must be either cpu, mps or cuda. '
                                                   f'Other devices are not tested/supported. Got: {args.device}.')

    if args.device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    segmentor = iSeekSegmentor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=not args.disable_progress_bar)

    segmentor.initialize_from_trained_model_folder(model_folder, args.f, checkpoint_name=args.chk)
    segmentor.predict_from_files(args.i, args.o,
                                 save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=args.num_parts,
                                 part_id=args.part_id)

    # Post
    dataset_json = join(args.o, "dataset.json")
    plans_json = join(args.o, "plans.json")
    args_json = join(args.o, "predict_from_raw_data_args.json")
    if os.path.exists(dataset_json):
        os.remove(dataset_json)
    if os.path.exists(plans_json):
        os.remove(plans_json)
    if os.path.exists(args_json):
        os.remove(args_json)

    # Do you want to run the evaluation?
    if args.eval:
        if args.gt is None:
            raise ValueError(
                "You need to specify the ground truth directory when you want to perform evaluation after segmentation")
        else:
            evaluation_after_segmentation(
                pred_dir=args.o,
                target_dir=args.gt,
                exclude_background=True,
                num_classes=None,
                workers=os.cpu_count()
            )
