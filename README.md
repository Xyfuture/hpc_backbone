# 软工项目 HPC AI Serve

使用模型：
- [lama](https://github.com/saic-mdal/lama)
- [DebulrGANv2](https://github.com/VITA-Group/DeblurGANv2)

权重文件需要预先下载
- [lama](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt)
- [DebulrGANv2](https://drive.google.com/uc?export=view&id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR)

建议下载到 `models/xxx/weight/`文件夹下

需要提前安装redis，并启动Web Serve

AI Serve启动方式:
`python run.py`