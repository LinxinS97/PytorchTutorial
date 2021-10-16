# fackbook专为pytorch开发的一款可视化工具
# https://github.com/fossasia/visdom
# 支持数值（折线图，直方图等），图像，文本以及视频等等
# 支持pytorch, torch和numpy
# 用户可以通过编程的方式组织可视化空间或者通过用户接口为数据打造仪表盘，检查实验结果和调试代码
#    env:环境 & pane:窗格

# pip install visdom
# python -m visdom.server
import visdom
import numpy as np
vis = visdom.Visdom()
vis.text('hello world!')
vis.image(np.ones((3, 10, 10)))

