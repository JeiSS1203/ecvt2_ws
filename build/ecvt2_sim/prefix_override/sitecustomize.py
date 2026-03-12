import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jin/harco/ecvt2_ws/install/ecvt2_sim'
