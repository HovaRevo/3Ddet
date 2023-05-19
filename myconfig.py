_base_ = [
    '../checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
]

import warnings

warnings.filterwarnings("ignore")

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    persistent_workers=True,
    # train=dict(dataset=dict(ann_file='data/kitti/kitti_infos_val.pkl'),)
    test=dict(
        split='testing',
        ann_file='data/kitti/kitti_infos_test.pkl',
        pts_prefix='velodyne2',
    )
)

optimizer = dict(
    type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01)
lr_config = None
momentum_config = None

runner = dict(max_epochs=80)
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5)
log_config = dict(interval=5)

# load_from = 'checkpoint/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
