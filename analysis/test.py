import qlib
from qlib.config import REG_CN
from qlib.contrib.data.handler import Alpha360, Alpha158, AlphaSimpleCustom
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    
    qlib.init(provider_uri="./qlib_data/cn_data", region=REG_CN)
    instruments = ["sz302132"]
    handler = AlphaSimpleCustom(); 
    fields, names = handler.get_feature_config(); 
    
    print('Features generated successfully:', len(fields))

