# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs
from ...data.dataset import processor as processor_module
from inspect import getfullargspec
import numpy as np


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
    # 统一特征模板，确保所有股票池特征一致且无缺失
    FEATURE_TEMPLATES = {
        "unified": {
            "ma": {"windows": [5, 10, 20, 30, 60]},
            "macd": {},
            "rsi": {"window": 14},
            "obv": {},
            "bollinger": {"window": 20, "std_dev": 2},
            #"atr": {"window": 14},
            "stoch": {"k_window": 14, "d_window": 3},
            "volatility": {"window": 10},   # 统一用10日波动率
            "liquidity": {"window": 3},     # 统一用3日流动性
            "turnover": {"window": 3},      # 统一用3日换手率
            "cap_tier": {}                    # 市值分层
        }
    }
    def get_feature_config(self):
        """所有股票池统一使用unified模板，避免特征缺失"""
        template = self.FEATURE_TEMPLATES["unified"]
        fields, names = self._parse_config(template)
        # 特征覆盖率监控（如有历史特征）
        if hasattr(self, 'prev_feat_names') and self.prev_feat_names:
            coverage, common = FeatureAlignmentHelper.compute_coverage(self.prev_feat_names, names)
            print(f"特征迁移覆盖率: {coverage:.2%}, 公共特征数: {len(common)}")
        return fields, names
    
    def _parse_config(self, config: dict) -> tuple:
        """解析配置为QLib字段，所有涉及分母的表达式加Abs()+1e-6保护，减少特征缺失"""
        fields, names = [], []
        
        # MA特征
        if "ma" in config:
            for w in config["ma"]["windows"]:
                fields.append(f"Mean($close, {w})/(Abs($close)+1e-6)")
                names.append(f"MA{w}")
       
        # MACD因子
        if "macd" in config:
            MACD_EXP = '((EMA($close, 12) - EMA($close, 26))/(Abs($close)+1e-6) - EMA((EMA($close, 12) - EMA($close, 26))/(Abs($close)+1e-6), 9)/(Abs($close)+1e-6))'
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
                (Mean(
                    If($close < Ref($close, 1), Ref($close, 1) - $close, 0),
                    {rsi_window}
                ) + 1e-6)
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
            ) / (Mean($volume, {volume_window})+1e-6)
            '''
            fields += [OBV_EXP]
            names += ["OBV"]

        # Bollinger Bands 中心线与收盘价比值
        if "bollinger" in config:
            window = config["bollinger"].get("window", 20)
            std_dev = config["bollinger"].get("std_dev", 2)
            window = max(1, window)  # 确保窗口参数大于0
            BOLL_MID = f'Mean($close, {window})'
            BOLL_EXP = f'(({BOLL_MID} - $close) / ((Std($close, {window})+1e-6) * {std_dev}))'
            fields += [BOLL_EXP]
            names += ["BOLL"]

        # ATR 平均真实波幅
        if "atr" in config:
            atr_window = config["atr"].get("window", 14)
            atr_window = max(1, atr_window)  # 确保窗口参数大于0
            TR1 = '($high - $low)'
            TR2 = 'Abs($high - Ref($close, 1))'
            TR3 = 'Abs($low - Ref($close, 1))'
            ATR_EXP = f'(Mean(Max(Max({TR1}, {TR2}), {TR3}), {atr_window}) / (Abs($close)+1e-6))'
            fields += [ATR_EXP]
            names += ["ATR"]

        # Stochastic Oscillator (随机指标)
        if "stoch" in config:
            k_window = config["stoch"].get("k_window", 14)
            d_window = config["stoch"].get("d_window", 3)
            # 确保窗口参数大于0
            k_window = max(1, k_window)
            d_window = max(1, d_window)
            RSV = f'( $close - Min($low, {k_window}) ) / ( (Max($high, {k_window}) - Min($low, {k_window}))+1e-6 )'
            K_EXP = f'Mean({RSV}, {d_window})'
            fields += [K_EXP]
            names += ["STOCH_K"]
        
        # 波动率特征
        if "volatility" in config:
            w = config["volatility"]["window"]
            fields.append(f"Std($close, {w})/(Abs($close)+1e-6)")
            names.append(f"VOL{w}")
        
        # 流动性特征
        if "liquidity" in config:
            w = config["liquidity"]["window"]
            fields.append(f"Mean($volume, {w})+1e-6")
            names.append(f"LIQ{w}")
        
         # 换手率特征
        if "turnover" in config:
            window = config["turnover"].get("window", 5)
            window = max(1, window)
            TURNOVER_EXP = f"$volume / (Mean($volume, {window})+1e-6)"
            fields.append(TURNOVER_EXP)
            names.append(f"TURNOVER{window}")

        # 市值分层特征
        if "cap_tier" in config:
            window = config.get("cap_tier_window", 200)
            CAP_TIER_EXP = f"Rank($amount, {window}) / (Count($amount, {window})+1e-6)"
            fields.append(CAP_TIER_EXP)
            names.append("CAP_TIER")
        
        return fields, names
    
    @staticmethod
    def get_label_expression(
        instruments: str,
        return_days: int = 5,
        csi300_thresh: float = 0.02,
        csi500_thresh: float = 0.02,
        quantile: float = 0.7,
        multi_label: bool = False,
        risk_label: bool = False
    ) -> tuple:
        """
        动态生成兼容不同QLib版本的标签表达式，支持：
        1. 参数化阈值/分位数
        2. 多标签/多分类
        3. 支持风险标签
        4. 多标签+风险标签
        """
        base_expr = f"Ref($close, -{return_days})/$close - 1"
        quantile_window = 200  # 默认窗口长度
        if '__qlib_version__' in globals() and __qlib_version__ >= "0.9.6":
            quantile_expr = f"Quantile({base_expr}, {quantile_window}, qscore={quantile})"
        else:
            quantile_expr = f"Quantile({base_expr}, {quantile_window}, {quantile})"

        # 多标签+风险标签同时启用
        if multi_label and risk_label:
            high_label = f"{base_expr} > {quantile_expr}"
            low_label = f"{base_expr} < Quantile({base_expr}, {quantile_window}, qscore=0.3)"
            drawdown_expr = f"(Min(Ref($close, -1, {return_days}), $close) - $close) / $close"
            drawdown_label = f"{drawdown_expr} < -0.05"  # 5日最大回撤大于5%"
            return [high_label, low_label, drawdown_label], ["LABEL_HIGH", "LABEL_LOW", "LABEL_DRAWDOWN"]

        # 1. 单标签（默认）
        if not multi_label and not risk_label:
            if "csi300" in str(instruments).lower():
                return [f"{base_expr} > {csi300_thresh}"], ["LABEL0"]
            elif "csi500" in str(instruments).lower():
                return [f"{base_expr} > {csi500_thresh}"], ["LABEL0"]
            else:
                return [f"{base_expr} > {quantile_expr}"], ["LABEL0"]

        # 2. 多标签/多分类
        if multi_label and not risk_label:
            high_label = f"{base_expr} > {quantile_expr}"
            low_label = f"{base_expr} < Quantile({base_expr}, {quantile_window}, qscore=0.3)"
            return [high_label, low_label], ["LABEL_HIGH", "LABEL_LOW"]

        # 3. 风险标签（如最大回撤）
        if risk_label:
            drawdown_expr = f"(Min(Ref($close, -1, {return_days}), $close) - $close) / $close"
            profit_label = f"{base_expr} > {quantile_expr}"
            drawdown_label = f"{drawdown_expr} < -0.05"  # 5日最大回撤大于5%"
            return [profit_label, drawdown_label], ["LABEL_PROFIT", "LABEL_DRAWDOWN"]

        # 兜底：返回单标签
        if "csi300" in str(instruments).lower():
            return [f"{base_expr} > {csi300_thresh}"], ["LABEL0"]
        elif "csi500" in str(instruments).lower():
            return [f"{base_expr} > {csi500_thresh}"], ["LABEL0"]
        else:
            return [f"{base_expr} > {quantile_expr}"], ["LABEL0"]

    def get_label_config(self, return_days=5, csi300_thresh=0.02, csi500_thresh=0.02, quantile=0.7, multi_label=False, risk_label=False):
        """
        动态生成标签表达式，支持参数化和多标签/风险标签
        """
        return self.get_label_expression(
            self.instruments,
            return_days=return_days,
            csi300_thresh=csi300_thresh,
            csi500_thresh=csi500_thresh,
            quantile=quantile,
            multi_label=multi_label,
            risk_label=risk_label
        )

class FeatureAlignmentHelper:
    """特征对齐与迁移辅助工具"""
    @staticmethod
    def compute_coverage(old_features, new_features):
        common = set(old_features) & set(new_features)
        coverage = len(common) / max(1, len(new_features))
        return coverage, list(common)

    @staticmethod
    def align_features(old_features, new_features, old_values, fill_method='zero'):
        """
        对齐特征：
        - 新特征用0或均值初始化
        - 丢失特征用历史均值或0填充
        """
        aligned = []
        for feat in new_features:
            if feat in old_features:
                idx = old_features.index(feat)
                aligned.append(old_values[idx])
            else:
                aligned.append(0.0 if fill_method == 'zero' else np.mean(old_values))
        return np.array(aligned)

    @staticmethod
    def visualize_feature_distribution(feature_matrix, feature_names, logger=None):
        desc = {}
        for i, name in enumerate(feature_names):
            col = feature_matrix[:, i]
            desc[name] = {
                'mean': float(np.mean(col)),
                'std': float(np.std(col)),
                'min': float(np.min(col)),
                'max': float(np.max(col))
            }
        if logger:
            logger.info(f"特征分布统计: {desc}")
        return desc