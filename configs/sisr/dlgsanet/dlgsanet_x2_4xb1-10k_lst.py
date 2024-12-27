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
        type="DLGSANet",
        upscale=scale,
        in_chans=1,
        dim=90,
        groups=6,
        blocks=4,
        buildblock_type='sparseedge',
        window_size=7,
        idynamic_num_heads=6,
        idynamic_ffn_type='GDFN',
        idynamic_ffn_expansion_factor=2.0,
        idynamic=True,
        restormer_num_heads=6,
        restormer_ffn_type='GDFN',
        restormer_ffn_expansion_factor=2.0,
        tlc_flag=True,
        tlc_kernel=96,  # using tlc during validation
        activation='relu',
        body_norm=False,
        img_range=1.0,
        upsampler='pixelshuffledirect',
        input_resolution=(256, 256),
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
