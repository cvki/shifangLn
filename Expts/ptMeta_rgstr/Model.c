[2022-12-09 20:34:20] [INFO] Command executed: /d2/code/gitRes/Expts/ptMeta_rgstr/experiments/3DMatch/trainval.py
[2022-12-09 20:34:20] [INFO] Configs:
{
    "seed": 7351,
    "working_dir": "/d2/code/gitRes/Expts/ptMeta_rgstr/experiments/3DMatch",
    "root_dir": "/d2/code/gitRes/Expts/ptMeta_rgstr",
    "exp_name": "3DMatch",
    "output_dir": "/d2/code/gitRes/Expts/ptMeta_rgstr/output/3DMatch",
    "snapshot_dir": "/d2/code/gitRes/Expts/ptMeta_rgstr/output/3DMatch/snapshots",
    "log_dir": "/d2/code/gitRes/Expts/ptMeta_rgstr/output/3DMatch/logs",
    "event_dir": "/d2/code/gitRes/Expts/ptMeta_rgstr/output/3DMatch/events",
    "feature_dir": "/d2/code/gitRes/Expts/ptMeta_rgstr/output/3DMatch/features",
    "registration_dir": "/d2/code/gitRes/Expts/ptMeta_rgstr/output/3DMatch/registration",
    "data": {
        "dataset_root": "/d2/code/gitRes/Expts/ptMeta_rgstr/data/3DMatch"
    },
    "train": {
        "batch_size": 1,
        "num_workers": 0,
        "point_limit": 30000,
        "use_augmentation": true,
        "augmentation_noise": 0.005,
        "augmentation_rotation": 1.0
    },
    "test": {
        "batch_size": 1,
        "num_workers": 0,
        "point_limit": null
    },
    "eval": {
        "acceptance_overlap": 0.0,
        "acceptance_radius": 0.1,
        "inlier_ratio_threshold": 0.05,
        "rmse_threshold": 0.2,
        "rre_threshold": 15.0,
        "rte_threshold": 0.3
    },
    "ransac": {
        "distance_threshold": 0.05,
        "num_points": 3,
        "num_iterations": 1000
    },
    "optim": {
        "lr": 0.0001,
        "lr_decay": 0.95,
        "lr_decay_steps": 1,
        "weight_decay": 1e-06,
        "max_epoch": 40,
        "grad_acc_steps": 1
    },
    "model": {
        "ground_truth_matching_radius": 0.05,
        "num_points_in_patch": 64,
        "num_sinkhorn_iterations": 100
    },
    "coarse_matching": {
        "num_targets": 128,
        "overlap_threshold": 0.1,
        "num_correspondences": 256,
        "dual_normalization": true
    },
    "geotransformer": {
        "input_dim": 512,
        "hidden_dim": 256,
        "output_dim": 256,
        "num_heads": 4,
        "blocks": [
            "self",
            "cross",
            "self",
            "cross",
            "self",
            "cross"
        ],
        "sigma_d": 0.2,
        "sigma_a": 15,
        "angle_k": 3,
        "reduction_a": "max"
    },
    "fine_matching": {
        "topk": 3,
        "acceptance_radius": 0.1,
        "mutual": true,
        "confidence_threshold": 0.05,
        "use_dustbin": false,
        "use_global_score": false,
        "correspondence_threshold": 3,
        "correspondence_limit": null,
        "num_refinement_steps": 5
    },
    "coarse_loss": {
        "positive_margin": 0.1,
        "negative_margin": 1.4,
        "positive_optimal": 0.1,
        "negative_optimal": 1.4,
        "log_scale": 24,
        "positive_overlap": 0.1
    },
    "fine_loss": {
        "positive_radius": 0.05
    },
    "loss": {
        "weight_coarse_loss": 1.0,
        "weight_fine_loss": 1.0
    }
}
[2022-12-09 20:34:20] [INFO] Tensorboard is enabled. Write events to /d2/code/gitRes/Expts/ptMeta_rgstr/output/3DMatch/events.
[2022-12-09 20:34:20] [INFO] Using Single-GPU mode.
[2022-12-09 20:34:20] [INFO] Data loader created: 0.059s collapsed.
[2022-12-09 20:34:20] [INFO] Calibrate neighbors: None.ooooooo.
[2022-12-09 20:34:21] [INFO] radius: [[0.1, 0.2, 0.2], [0.2, 0.4, 0.4, 0.4, 0.4], [0.4, 0.8, 0.8], [0.8, 1.6, 1.6]],
 nsample: [[32, 32, 32], [32, 32, 32, 32, 32], [32, 32, 32], [32, 32, 32]]
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.1
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.2
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.2
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.2
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.4
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.4
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.4
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.4
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.4
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.4
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.8
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.8
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.8
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.8
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 1.6
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 1.6
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 1.6
nsample: 32
[2022-12-09 20:34:21] [INFO] radius: [[0.1, 0.2, 0.2], [0.2, 0.4, 0.4, 0.4, 0.4], [0.4, 0.8, 0.8], [0.8, 1.6, 1.6]],
 nsample: [[32, 32, 32], [32, 32, 32, 32, 32], [32, 32, 32], [32, 32, 32]]
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.1
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.2
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.2
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.2
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.4
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.4
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.4
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.4
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.4
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.4
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.8
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.8
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.8
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 0.8
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 1.6
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 1.6
nsample: 32
[2022-12-09 20:34:21] [INFO] NAME: ballquery
normalize_dp: True
radius: 1.6
nsample: 32
[2022-12-09 20:34:24] [INFO] Model description:
GeoTransformer(
  (backbone): BaseSeg(
    (encoder): PointMetaBaseEncoder(
      (encoder): Sequential(
        (0): Sequential(
          (0): SetAbstraction(
            (convs1): Sequential(
              (0): Sequential(
                (0): Conv1d(4, 64, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (convs2): Sequential(
              (0): Sequential(
                (0): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (grouper): QueryAndGroup()
          )
          (1): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
          (2): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
        )
        (1): Sequential(
          (0): SetAbstraction(
            (convs1): Sequential(
              (0): Sequential(
                (0): Conv1d(64, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (convs2): Sequential(
              (0): Sequential(
                (0): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (grouper): QueryAndGroup()
          )
          (1): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
          (2): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
          (3): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
          (4): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
        )
        (2): Sequential(
          (0): SetAbstraction(
            (convs1): Sequential(
              (0): Sequential(
                (0): Conv1d(128, 256, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (convs2): Sequential(
              (0): Sequential(
                (0): Conv2d(3, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (grouper): QueryAndGroup()
          )
          (1): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
          (2): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
        )
        (3): Sequential(
          (0): SetAbstraction(
            (convs1): Sequential(
              (0): Sequential(
                (0): Conv1d(256, 512, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (convs2): Sequential(
              (0): Sequential(
                (0): Conv2d(3, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (grouper): QueryAndGroup()
          )
          (1): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
          (2): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
        )
      )
      (pe_encoder): ModuleList(
        (0): ModuleList()
        (1): Sequential(
          (0): Sequential(
            (0): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
        )
        (2): Sequential(
          (0): Sequential(
            (0): Conv2d(3, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
        )
        (3): Sequential(
          (0): Sequential(
            (0): Conv2d(3, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
        )
      )
    )
    (decoder): PointMetaBaseEncoder(
      (encoder): Sequential(
        (0): Sequential(
          (0): SetAbstraction(
            (convs1): Sequential(
              (0): Sequential(
                (0): Conv1d(4, 64, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (convs2): Sequential(
              (0): Sequential(
                (0): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (grouper): QueryAndGroup()
          )
          (1): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
          (2): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
        )
        (1): Sequential(
          (0): SetAbstraction(
            (convs1): Sequential(
              (0): Sequential(
                (0): Conv1d(64, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (convs2): Sequential(
              (0): Sequential(
                (0): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (grouper): QueryAndGroup()
          )
          (1): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
          (2): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
          (3): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
          (4): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
        )
        (2): Sequential(
          (0): SetAbstraction(
            (convs1): Sequential(
              (0): Sequential(
                (0): Conv1d(128, 256, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (convs2): Sequential(
              (0): Sequential(
                (0): Conv2d(3, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (grouper): QueryAndGroup()
          )
          (1): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
          (2): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
        )
        (3): Sequential(
          (0): SetAbstraction(
            (convs1): Sequential(
              (0): Sequential(
                (0): Conv1d(256, 512, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (convs2): Sequential(
              (0): Sequential(
                (0): Conv2d(3, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
            )
            (grouper): QueryAndGroup()
          )
          (1): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
          (2): InvResMLP(
            (convs): LocalAggregation(
              (convs1): Sequential(
                (0): Sequential(
                  (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,), bias=False)
                  (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU(inplace=True)
                )
              )
              (grouper): QueryAndGroup()
            )
            (pwconv): Sequential(
              (0): Sequential(
                (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
              )
              (1): Sequential(
                (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act): ReLU(inplace=True)
          )
        )
      )
      (pe_encoder): ModuleList(
        (0): ModuleList()
        (1): Sequential(
          (0): Sequential(
            (0): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
        )
        (2): Sequential(
          (0): Sequential(
            (0): Conv2d(3, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
        )
        (3): Sequential(
          (0): Sequential(
            (0): Conv2d(3, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
        )
      )
    )
  )
  (transformer): GeometricTransformer(
    (embedding): GeometricStructureEmbedding(
      (embedding): SinusoidalPositionalEmbedding()
      (proj_d): Linear(in_features=256, out_features=256, bias=True)
      (proj_a): Linear(in_features=256, out_features=256, bias=True)
    )
    (in_proj): Linear(in_features=512, out_features=256, bias=True)
    (transformer): RPEConditionalTransformer(
      (layers): ModuleList(
        (0): RPETransformerLayer(
          (attention): RPEAttentionLayer(
            (attention): RPEMultiHeadAttention(
              (proj_q): Linear(in_features=256, out_features=256, bias=True)
              (proj_k): Linear(in_features=256, out_features=256, bias=True)
              (proj_v): Linear(in_features=256, out_features=256, bias=True)
              (proj_p): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Identity()
            )
            (linear): Linear(in_features=256, out_features=256, bias=True)
            (dropout): Identity()
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
          (output): AttentionOutput(
            (expand): Linear(in_features=256, out_features=512, bias=True)
            (activation): ReLU()
            (squeeze): Linear(in_features=512, out_features=256, bias=True)
            (dropout): Identity()
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
        (1): TransformerLayer(
          (attention): AttentionLayer(
            (attention): MultiHeadAttention(
              (proj_q): Linear(in_features=256, out_features=256, bias=True)
              (proj_k): Linear(in_features=256, out_features=256, bias=True)
              (proj_v): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Identity()
            )
            (linear): Linear(in_features=256, out_features=256, bias=True)
            (dropout): Identity()
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
          (output): AttentionOutput(
            (expand): Linear(in_features=256, out_features=512, bias=True)
            (activation): ReLU()
            (squeeze): Linear(in_features=512, out_features=256, bias=True)
            (dropout): Identity()
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
        (2): RPETransformerLayer(
          (attention): RPEAttentionLayer(
            (attention): RPEMultiHeadAttention(
              (proj_q): Linear(in_features=256, out_features=256, bias=True)
              (proj_k): Linear(in_features=256, out_features=256, bias=True)
              (proj_v): Linear(in_features=256, out_features=256, bias=True)
              (proj_p): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Identity()
            )
            (linear): Linear(in_features=256, out_features=256, bias=True)
            (dropout): Identity()
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
          (output): AttentionOutput(
            (expand): Linear(in_features=256, out_features=512, bias=True)
            (activation): ReLU()
            (squeeze): Linear(in_features=512, out_features=256, bias=True)
            (dropout): Identity()
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
        (3): TransformerLayer(
          (attention): AttentionLayer(
            (attention): MultiHeadAttention(
              (proj_q): Linear(in_features=256, out_features=256, bias=True)
              (proj_k): Linear(in_features=256, out_features=256, bias=True)
              (proj_v): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Identity()
            )
            (linear): Linear(in_features=256, out_features=256, bias=True)
            (dropout): Identity()
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
          (output): AttentionOutput(
            (expand): Linear(in_features=256, out_features=512, bias=True)
            (activation): ReLU()
            (squeeze): Linear(in_features=512, out_features=256, bias=True)
            (dropout): Identity()
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
        (4): RPETransformerLayer(
          (attention): RPEAttentionLayer(
            (attention): RPEMultiHeadAttention(
              (proj_q): Linear(in_features=256, out_features=256, bias=True)
              (proj_k): Linear(in_features=256, out_features=256, bias=True)
              (proj_v): Linear(in_features=256, out_features=256, bias=True)
              (proj_p): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Identity()
            )
            (linear): Linear(in_features=256, out_features=256, bias=True)
            (dropout): Identity()
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
          (output): AttentionOutput(
            (expand): Linear(in_features=256, out_features=512, bias=True)
            (activation): ReLU()
            (squeeze): Linear(in_features=512, out_features=256, bias=True)
            (dropout): Identity()
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
        (5): TransformerLayer(
          (attention): AttentionLayer(
            (attention): MultiHeadAttention(
              (proj_q): Linear(in_features=256, out_features=256, bias=True)
              (proj_k): Linear(in_features=256, out_features=256, bias=True)
              (proj_v): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Identity()
            )
            (linear): Linear(in_features=256, out_features=256, bias=True)
            (dropout): Identity()
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
          (output): AttentionOutput(
            (expand): Linear(in_features=256, out_features=512, bias=True)
            (activation): ReLU()
            (squeeze): Linear(in_features=512, out_features=256, bias=True)
            (dropout): Identity()
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
    )
    (out_proj): Linear(in_features=256, out_features=256, bias=True)
  )
  (coarse_target): SuperPointTargetGenerator()
  (coarse_matching): SuperPointMatching()
  (fine_matching): LocalGlobalRegistration(
    (procrustes): WeightedProcrustes()
  )
  (optimal_transport): LearnableLogOptimalTransport(num_iterations=100)
)