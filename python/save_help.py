import os
import sys
import re
from multiprocessing import Process

# 重定向sys.stdout
old_stdout = sys.stdout

# 获取所有模块
f = open('modules.txt', 'r')
modules = f.read().split()
f.close()

for module in modules:
    sys.stdout = open(f'modules\\{module}.txt', 'w')
    help(module)

# 还原sys.stdout
sys.stdout.close()
sys.stdout = old_stdout
del old_stdout
