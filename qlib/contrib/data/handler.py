# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs
from ...data.dataset import processor as processor_module
from inspect import getfullargspec


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class Alpha360(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": Alpha360DL.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs,
        )

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha360vwap(Alpha360):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]


class Alpha158(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        # 先设置instruments属性，以便在get_feature_config中使用
        self.instruments = instruments
        
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        # print("AlphaSimpleCustom config:", data_loader)
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [max(1, w) for w in [1]],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        return Alpha158DL.get_feature_config(conf)

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]

class AlphaSimpleCustom(Alpha158):
    def get_feature_config(self):
        # 8个因子：5/10/20/30/60日均线比值，macd，rsi，obv，布林带，atr，stoch
        conf = {
            "ma": {"windows": [max(1, w) for w in [5, 10, 20, 30, 60]]},
            "macd": {},
            "rsi": {"window": max(1, 14)},
            "obv": {},
            "bollinger": {"window": max(1, 20), "std_dev": 2},
            #"atr": {"window": max(1, 14)},
            "stoch": {"k_window": max(1, 14), "d_window": max(1, 3)}
        }
        return self.parse_config_to_fields(conf)

    @staticmethod
    def parse_config_to_fields(config):
        fields = []
        names = []

        # 移动平均线因子
        if "ma" in config:
            windows = config["ma"].get("windows", [])
            fields += ["Mean($close, %d)/$close" % d for d in windows]
            names += ["MA%d" % d for d in windows]

        # MACD因子
        if "macd" in config:
            MACD_EXP = '(EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)/$close'
            fields += [MACD_EXP]
            names += ["MACD"]

        # RSI相对强弱指数
        if "rsi" in config:
            rsi_window = config["rsi"].get("window", 14)
            rsi_window = max(1, rsi_window)  # 确保窗口参数大于0
            RSI_EXP = f'''
            100 - (100 / (1 + (
                Mean(
                    If($close > Ref($close, 1), $close - Ref($close, 1), 0),
                    {rsi_window}
                ) /
                Mean(
                    If($close < Ref($close, 1), Ref($close, 1) - $close, 0),
                    {rsi_window}
                )
            )))
            '''
            fields += [RSI_EXP]
            names += ["RSI"]

        # OBV能量潮指标
        if "obv" in config:
            obv_window = config["obv"].get("window", 20)
            volume_window = config["obv"].get("volume_window", 20)
            obv_window = max(1, obv_window)
            volume_window = max(1, volume_window)

            OBV_EXP = f'''
            Sum(
                If($close > Ref($close, 1), $volume,
                If($close < Ref($close, 1), 0-$volume, 0)),
                {obv_window}
            ) / Mean($volume, {volume_window})
            '''
            fields += [OBV_EXP]
            names += ["OBV"]

        # Bollinger Bands 中心线与收盘价比值
        if "bollinger" in config:
            window = config["bollinger"].get("window", 20)
            std_dev = config["bollinger"].get("std_dev", 2)
            window = max(1, window)  # 确保窗口参数大于0
            BOLL_MID = f'Mean($close, {window})'
            BOLL_EXP = f'({BOLL_MID} - $close) / (Std($close, {window}) * {std_dev})'
            fields += [BOLL_EXP]
            names += ["BOLL"]

        # ATR 平均真实波幅
        if "atr" in config:
            atr_window = config["atr"].get("window", 14)
            atr_window = max(1, atr_window)  # 确保窗口参数大于0
            print("ATR window:", atr_window)
            # ATR = Mean(True Range, window), True Range = Max(H-L, |H-C_prev|, |L-C_prev|)
            TR1 = '($high - $low)'
            TR2 = 'Abs($high - Ref($close, 1))'
            TR3 = 'Abs($low - Ref($close, 1))'
            ATR_EXP = f'Mean(Max(Max({TR1}, {TR2}), {TR3}), {atr_window}) / $close'
            print("ATR exp:", ATR_EXP)
            fields += [ATR_EXP]
            names += ["ATR"]

        # Stochastic Oscillator (随机指标)
        if "stoch" in config:
            k_window = config["stoch"].get("k_window", 14)
            d_window = config["stoch"].get("d_window", 3)
            # 确保窗口参数大于0
            k_window = max(1, k_window)
            d_window = max(1, d_window)
            RSV = f'( $close - Min($low, {k_window}) ) / ( Max($high, {k_window}) - Min($low, {k_window}) )'
            K_EXP = f'Mean({RSV}, {d_window})'
            fields += [K_EXP]
            names += ["STOCH_K"]

        return fields, names

from qlib import __version__ as __qlib_version__

class DynamicAlphaCustom(Alpha158):
    """动态适配不同股票池的特征处理器"""
    def __init__(self, **kwargs):
        self._check_qlib_version()
        super().__init__(**kwargs)
    
    def _check_qlib_version(self):
        """验证QLib版本兼容性"""
        from distutils.version import LooseVersion
        if LooseVersion(__qlib_version__) < LooseVersion("0.9.6"):
            logger.warning(f"QLib {__qlib_version__} may have different API signatures")
    # 不同股票池的特征配置模板
    FEATURE_TEMPLATES = {
        "default": {
            "ma": {"windows": [max(1, w) for w in [5, 10, 20, 30, 60]]},
            "macd": {},
            "rsi": {"window": max(1, 14)},
            "obv": {},
            "bollinger": {"window": max(1, 20), "std_dev": 2},
            #"atr": {"window": max(1, 14)},
            "stoch": {"k_window": max(1, 14), "d_window": max(1, 3)}
        },
        "csi300": {
            "ma": {"windows": [max(1, w) for w in [5, 10, 20, 30, 60]]},
            "macd": {},
            "rsi": {"window": max(1, 14)},
            "obv": {},
            "bollinger": {"window": max(1, 20), "std_dev": 2},
            #"atr": {"window": max(1, 14)},
            "stoch": {"k_window": max(1, 14), "d_window": max(1, 3)}
        },
        "csi500": {
            "ma": {"windows": [max(1, w) for w in [5, 10, 20, 30, 60]]},
            "macd": {},
            "rsi": {"window": max(1, 14)},
            "obv": {},
            "bollinger": {"window": max(1, 20), "std_dev": 2},
            #"atr": {"window": max(1, 14)},
            "stoch": {"k_window": max(1, 14), "d_window": max(1, 3)},
            "volatility": {"window": 20}  # 新增波动率特征
        },
        "csi800": {
            "ma": {"windows": [max(1, w) for w in [5, 10, 20, 30, 60]]},
            "macd": {},
            "rsi": {"window": max(1, 14)},
            "obv": {},
            "bollinger": {"window": max(1, 20), "std_dev": 2},
            #"atr": {"window": max(1, 14)},
            "stoch": {"k_window": max(1, 14), "d_window": max(1, 3)},
            "liquidity": {"window": 5}  # 流动性特征
        },
        "all": {
            "ma": {"windows": [max(1, w) for w in [5, 10, 20, 30, 60]]},
            "macd": {},
            "rsi": {"window": max(1, 14)},
            "obv": {},
            "bollinger": {"window": max(1, 20), "std_dev": 2},
            #"atr": {"window": max(1, 14)},
            "stoch": {"k_window": max(1, 14), "d_window": max(1, 3)},
            "cap_tier": {}  # 市值分层特征
        }
    }    
    def get_feature_config(self) -> tuple:
        """动态选择特征配置"""
        if "csi300" in self.instruments:
            template = self.FEATURE_TEMPLATES["csi300"]
        elif "csi500" in self.instruments:
            template = self.FEATURE_TEMPLATES["csi500"]
        elif "csi800" in self.instruments:
            template = self.FEATURE_TEMPLATES["csi800"]
        elif "all" in self.instruments:
            template = self.FEATURE_TEMPLATES["all"]
        else:
            template = self.FEATURE_TEMPLATES["default"]
        
        return self._parse_config(template)
    
    def _parse_config(self, config: dict) -> tuple:
        """解析配置为QLib字段"""
        fields, names = [], []
        
        # MA特征
        if "ma" in config:
            for w in config["ma"]["windows"]:
                fields.append(f"Mean($close, {w})/$close")
                names.append(f"MA{w}")
       
        # MACD因子
        if "macd" in config:
            MACD_EXP = '(EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)/$close'
            fields += [MACD_EXP]
            names += ["MACD"]

        # RSI相对强弱指数
        if "rsi" in config:
            rsi_window = config["rsi"].get("window", 14)
            rsi_window = max(1, rsi_window)  # 确保窗口参数大于0
            RSI_EXP = f'''
            100 - (100 / (1 + (
                Mean(
                    If($close > Ref($close, 1), $close - Ref($close, 1), 0),
                    {rsi_window}
                ) /
                Mean(
                    If($close < Ref($close, 1), Ref($close, 1) - $close, 0),
                    {rsi_window}
                )
            )))
            '''
            fields += [RSI_EXP]
            names += ["RSI"]

        # OBV能量潮指标
        if "obv" in config:
            obv_window = config["obv"].get("window", 20)
            volume_window = config["obv"].get("volume_window", 20)
            obv_window = max(1, obv_window)
            volume_window = max(1, volume_window)

            OBV_EXP = f'''
            Sum(
                If($close > Ref($close, 1), $volume,
                If($close < Ref($close, 1), 0-$volume, 0)),
                {obv_window}
            ) / Mean($volume, {volume_window})
            '''
            fields += [OBV_EXP]
            names += ["OBV"]

        # Bollinger Bands 中心线与收盘价比值
        if "bollinger" in config:
            window = config["bollinger"].get("window", 20)
            std_dev = config["bollinger"].get("std_dev", 2)
            window = max(1, window)  # 确保窗口参数大于0
            BOLL_MID = f'Mean($close, {window})'
            BOLL_EXP = f'({BOLL_MID} - $close) / (Std($close, {window}) * {std_dev})'
            fields += [BOLL_EXP]
            names += ["BOLL"]

        # ATR 平均真实波幅
        if "atr" in config:
            atr_window = config["atr"].get("window", 14)
            atr_window = max(1, atr_window)  # 确保窗口参数大于0
            print("ATR window:", atr_window)
            # ATR = Mean(True Range, window), True Range = Max(H-L, |H-C_prev|, |L-C_prev|)
            TR1 = '($high - $low)'
            TR2 = 'Abs($high - Ref($close, 1))'
            TR3 = 'Abs($low - Ref($close, 1))'
            ATR_EXP = f'Mean(Max(Max({TR1}, {TR2}), {TR3}), {atr_window}) / $close'
            print("ATR exp:", ATR_EXP)
            fields += [ATR_EXP]
            names += ["ATR"]

        # Stochastic Oscillator (随机指标)
        if "stoch" in config:
            k_window = config["stoch"].get("k_window", 14)
            d_window = config["stoch"].get("d_window", 3)
            # 确保窗口参数大于0
            k_window = max(1, k_window)
            d_window = max(1, d_window)
            RSV = f'( $close - Min($low, {k_window}) ) / ( Max($high, {k_window}) - Min($low, {k_window}) )'
            K_EXP = f'Mean({RSV}, {d_window})'
            fields += [K_EXP]
            names += ["STOCH_K"]
        
        # 波动率特征 (中盘股专用)
        if "volatility" in config and ("csi500" in self.instruments or "csi800" in self.instruments):
            w = config["volatility"]["window"]
            fields.append(f"Std($close, {w})/$close")
            names.append(f"VOL{w}")
        
        # 流动性特征 (全A股专用)
        if "liquidity" in config and ("csi800" in self.instruments or "all" in self.instruments):
            w = config["liquidity"]["window"]
            fields.append(f"Mean($volume, {w})")
            names.append(f"LIQ{w}")
        
        return fields, names
    
    def get_label_expression(instruments: str) -> tuple:
        """动态生成兼容不同QLib版本的标签表达式"""
        base_expr = "Ref($close, -5)/$close - 1"
        
        if __qlib_version__ >= "0.9.6":
            quantile_expr = f"Quantile({base_expr}, qscore=0.7)"
        else:
            quantile_expr = f"Quantile({base_expr}, 0.7)"
        
        if "csi300" in instruments.lower():
            return [f"{base_expr} > 0.02"], ["LABEL0"]
        elif "csi500" in instruments.lower():
            return [f"{base_expr} > 0.03"], ["LABEL0"]
        else:
            return [f"{base_expr} > {quantile_expr}"], ["LABEL0"]