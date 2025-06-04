import unittest
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import qlib
from qlib.contrib.data.handler import AlphaSimpleCustom
from qlib.config import REG_CN

class TestAlphaSimpleCustom(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # 使用项目根目录下的数据路径
        data_path = os.path.join(project_root, "qlib_data", "cn_data")
        qlib.init(provider_uri=data_path, region=REG_CN)
    
    def test_feature_config(self):
        """测试特征配置"""
        handler = AlphaSimpleCustom(
            start_time="2020-01-01",
            end_time="2020-12-31",
            instruments="csi300"
        )
        
        fields, names = handler.get_feature_config()
        
        # 验证特征数量
        self.assertGreater(len(fields), 0)
        self.assertEqual(len(fields), len(names))
        
        # 验证特征名称
        expected_features = ['MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'MACD', 'RSI', 'OBV']
        for feature in expected_features:
            self.assertIn(feature, names)
    


    def test_data_loading(self):
        """测试数据加载"""
        handler = AlphaSimpleCustom(
            start_time="2020-01-01",
            end_time="2020-01-31",
            instruments="csi300"
        )
        
        # 验证数据可以正常加载
        self.assertIsNotNone(handler._data)

if __name__ == '__main__':
    unittest.main()