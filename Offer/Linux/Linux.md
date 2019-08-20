## linux常用命令
- watch -n 5 nvidia-smi  # 查看显卡信息，每隔五秒刷新一次显示
- find：这个命令用于查找文件，功能强大。例如：find ./ -name "*.md"：查找当前目录及其子目录下所有扩展名是 .md 的文件。
- kill PID：杀死某进程。kill -s 9 PID：强制杀死进程
- du -h或du -h --max-depth=1：linux中查看各文件夹及其子文件夹大小命令，后者以当前目录为节点，只往目录树下查找一层，即当前目录下的文件夹（不包括子文件夹）。
- ls | wc -w :查看当前目录下有多少个文件及文件夹（不包括子文件夹）

### 推荐/参考链接
- [深度学习中常用的linux命令](https://blog.csdn.net/ft_sunshine/article/details/91993590)
- [linux 常用的 20 个命令](https://blog.csdn.net/q357010621/article/details/80248611)
- [linux 命令大全](https://www.runoob.com/linux/linux-command-manual.html)
