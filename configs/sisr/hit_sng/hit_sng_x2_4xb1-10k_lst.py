_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/datasets/heihe_lst/heihe_lst_x2-256-512_sisr.py",
    "../../_base_/schedules/schedule_10k.py",
]
scale = 2
# model settings
model = dict(
    type="BaseEditModel",
    generator=dict(
        type="HiT_SNG",
        # https://github.com/XiangZ-0/HiT-SR/blob/main/options/Train/train_HiT_SIR_x2.yml        
        img_size=256,
        patch_size=1,
        in_chans=1,
        embed_dim=60,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        base_win_size=[8, 8],
        mlp_ratio=2.0,
        drop_rate=0.0,
        value_drop_rate=0.0,
        drop_path_rate=0.0,
        # norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=scale,
        img_range=1.0,
        upsampler='pixelshuffledirect',
        resi_connection='1conv',
        hier_win_ratios=[0.5, 1, 2, 4, 6, 8],
        norm_flag=1,  # 0: not norm; 1: z-score; 2: min-max norm; other number: assert error.
        norm_dict=dict(mean=282.51, std=14.73, min=242.50, max=322.33),  # 60m norm paras
    ),
    pixel_loss=dict(type="L1Loss", loss_weight=1.0, reduction="mean"),
    train_cfg=dict(),
    test_cfg=dict(
        metrics=[
            dict(type="RMSE", scaling=1.0, prefix="lst"),
            dict(type="LSTMAE", scaling=1.0, prefix="lst"),  # abs(pred-gt)
            dict(type="BIAS", scaling=1.0, prefix="lst"),  # pred-gt
            dict(type="CC", scaling=1.0, prefix="lst"),
            dict(type="RSD", scaling=1.0, prefix="lst"),
        ]
    ),
    data_preprocessor=dict(  # TODO
        type="LSTDataPreprocessor",
        mean=None,
        std=None,
    ),
)
