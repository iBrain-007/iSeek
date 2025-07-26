import pydoc
from typing import Tuple, Union, List
import numpy as np
import torch
from torch import autocast, nn
from torch._dynamo import OptimizedModule

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from iseek.core.losses.pixel_local_connectivity_loss import PixelLocalConnectivityLoss
from iseek.core.losses.to_neighbor_connectivity import to_nk_maps
from iseek.core.models.segmentation.ultra_framework import Ultra

torch.set_float32_matmul_precision('high')


class UltraS3Trainer(nnUNetTrainer):
    def __init__(self,
                 plans: dict,
                 configuration: str,
                 fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-2  # maybe 3e-4 is better than 1e-2 for AdamW optimizer
        self.num_epochs = 1000  # Do you want to wait longer? Increase it.
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50

        self.enable_deep_supervision = True

        # Adjust Your Optimization Objectives Here!
        self.S = 3
        self.nk_kernels = [2 * (s + 1) + 1 for s in range(self.S)]
        self.nk_masking = False
        self.w1 = 1.0
        self.w2 = 2.0  # give more important to stage2 than stage1
        self.lambda_plc = 1.0
        # we define the vessel local connectivity loss
        self.loss_s1, self.plc = self._build_plc_loss()
        self.print_to_log_file([], also_print_to_console=False, add_timestamp=False)

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # ### CAUTION: change the scale number of the neighborhood scale here.
        neighborhood_scale = 3  # This should be the same as self.S

        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
        n_stages = architecture_kwargs['n_stages'] - 1

        coarse_model = nnUNetTrainer.build_network_architecture(
            architecture_class_name=architecture_class_name,
            arch_init_kwargs=arch_init_kwargs,
            arch_init_kwargs_req_import=arch_init_kwargs_req_import,
            num_input_channels=num_input_channels,
            num_output_channels=1,  # always binary segmentation
            enable_deep_supervision=enable_deep_supervision
        )
        coarse_model.decoder.deep_supervision = True  # to obtain multi-scale prior masks
        network = Ultra(
            coarse_model=coarse_model,
            coarse_out_channels=1,  # the coarse model is always a binary segmentation backbone
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            num_pool=n_stages,
            neighbor_scale=neighborhood_scale,
            hidden_dims=[32, 32, 32],
            dropout_rate=0.0,
            deep_supervision=enable_deep_supervision,
            is_training=enable_deep_supervision,
        )
        return network

    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.coarse_model.decoder.deep_supervision = True  # Must be True because we want to get it all the time.
        mod.deep_supervision = enabled
        mod.is_training = enabled

    def _build_plc_loss(self):
        loss_bin = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        plc_loss = PixelLocalConnectivityLoss(
            {},
            {'batch_dice': self.configuration_manager.batch_dice,
             'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
            weight_ce=1.0,
            weight_dice=1.0,  # disable Dice loss, enable the BCE loss only.
            use_ignore_label=self.label_manager.ignore_label is not None,
            dice_class=MemoryEfficientSoftDiceLoss,
            nk_masking=self.nk_masking)

        if self._do_i_compile():
            loss_bin.dc = torch.compile(loss_bin.dc)
            plc_loss.dc = torch.compile(plc_loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()

            loss_bin = DeepSupervisionWrapper(loss_bin, weights)
            plc_loss = DeepSupervisionWrapper(plc_loss, weights)

        return loss_bin, plc_loss

    def train_step(self, batch: dict) -> dict:
        """
        Our train step based on nnUNetv2 framework.
        """
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            bin_target = [(i > 0).float() for i in target]  # for stage1: PriorNet
            nk_maps, nk_masks = to_nk_maps(target, kernel_sizes=self.nk_kernels)  # [B, total_neighbors, H*W]
            nk_maps = [i.to(self.device, non_blocking=True) for i in nk_maps]
            nk_masks = [i.to(self.device, non_blocking=True) for i in nk_masks]  # Using masking is better?
        else:
            target = target.to(self.device, non_blocking=True)
            bin_target = (target > 0).float()  # for stage1: PriorNet
            nk_maps, nk_masks = to_nk_maps(target, kernel_sizes=self.nk_kernels)  # [B, total_neighbors, H*W]
            nk_maps = nk_maps.to(self.device, non_blocking=True)
            nk_masks = nk_masks.to(self.device, non_blocking=True)  # Using masking is better?

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            coarse, output, nk = self.network(data)

            target = target[:len(output)] if isinstance(target, list) else target
            nk_maps = nk_maps[:len(output)] if isinstance(nk_maps, list) else nk_maps

            l_s1 = self.loss_s1(coarse, bin_target)  # binary segmentation
            l_s2 = self.loss(output, target)
            l_plc = self.plc(nk, nk_maps)
            l = self.w1 * l_s1 + self.w2 * l_s2 + self.lambda_plc * l_plc

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            bin_target = [(i > 0).float() for i in target]  # for stage1: PriorNet
            nk_maps, nk_masks = to_nk_maps(target, kernel_sizes=self.nk_kernels)  # [B, total_neighbors, H*W]
            nk_maps = [i.to(self.device, non_blocking=True) for i in nk_maps]
            nk_masks = [i.to(self.device, non_blocking=True) for i in nk_masks]  # Using masking is better?
        else:
            target = target.to(self.device, non_blocking=True)
            bin_target = (target > 0).float()  # for stage1: PriorNet
            nk_maps, nk_masks = to_nk_maps(target, kernel_sizes=self.nk_kernels)  # [B, total_neighbors, H*W]
            nk_maps = nk_maps.to(self.device, non_blocking=True)
            nk_masks = nk_masks.to(self.device, non_blocking=True)  # Using masking is better?

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            coarse, output, nk = self.network(data)
            del data
            target = target[:len(output)] if isinstance(target, list) else target
            nk_maps = nk_maps[:len(output)] if isinstance(nk_maps, list) else nk_maps

            l_s1 = self.loss_s1(coarse, bin_target)  # binary segmentation
            l_s2 = self.loss(output, target)
            l_plc = self.plc(nk, nk_maps)
            l = self.w1 * l_s1 + self.w2 * l_s2 + self.lambda_plc * l_plc

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]
        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.5,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.3
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)
