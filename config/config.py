# -*- coding: utf-8 -*-
config = {
    "gpu": "7",
    "seed" : 222,
    "epochs" : 600,
    "threshold" : 0.3,
    "input_dim": 15,
    "num_classes1": 7,
    "num_classes2": 22,
    "num_classes3": 135,
    "dim": 128,
    "heads": 4,
    "mlp_dim": 256,
    "dim_head": 32,
    "dropout": 0.,
    "trans_depth" : 2,
    "outlier_depth" : 2,
    "SAVE_DIR_TEMPLATE" : "checkpoints/",
    }

DATA_PATHS = {
    "train": "",
    "val": "",
    "test": "",
    "top_train": "",
    "top_test": "/",
    "top_val":"",
}