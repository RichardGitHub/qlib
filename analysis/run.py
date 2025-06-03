import qlib
from qlib.constant import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
import logging
from multiprocessing import freeze_support
from qlib.tests.data import GetData
from qlib.data import D

logger = logging.getLogger(__name__)

provider_uri = "./qlib_data/cn_data"


market = "csi300"
benchmark = "SH000300"

data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2025-05-29",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2025-05-29",
    "instruments": market,
}

task = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2008-01-01", "2018-12-31"),
                "valid": ("2019-01-01", "2021-12-31"),
                "test": ("2022-01-01", "2023-12-31"),
            },
        },
    },
}

if __name__ == "__main__":
    
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    
    freeze_support()
    try:
        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])
        
        with R.start(experiment_name="workflow"):
            R.log_params(**flatten_dict(task))
            model.fit(dataset)
            
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise