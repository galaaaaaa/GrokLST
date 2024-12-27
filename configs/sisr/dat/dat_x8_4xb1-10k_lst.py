_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/datasets/heihe_lst/heihe_lst_x8-64-512_sisr.py",
    "../../_base_/schedules/schedule_10k.py",
]

scale = 8
# model settings
model = dict(
    type="BaseEditModel",
    generator=dict(
        type="DAT",
        # https://github.com/zhengchen1999/DAT/blob/main/options/Train/train_DAT_x2.yml
        upscale=scale,
        in_chans=1,
        img_size=64,
        img_range=1.,
        split_size=[8,32],
        depth=[6,6,6,6,6,6],
        embed_dim=180,
        num_heads=[6,6,6,6,6,6],
        expansion_factor=4,
        resi_connection='1conv',
        norm_flag=1,  # 0: not norm; 1: z-score; 2: min-max norm; other number: assert error.
        norm_dict=dict(mean=282.51, std=14.59, min=243.53, max=320.85),  # 240m paras
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
