#!/bin/bash -l

# 这里写上作业名称
#SBATCH --job-name=job_name           

# 分区名称，该参数必须写，咨询管理员可用分区名称
#SBATCH --partition=3090gpu                    ##作业申请的分区名称

# 申请节点数
#SBATCH --nodes=1                           ##作业申请的节点数

# 每个节点的核心数
#SBATCH --ntasks-per-node=10           ##作业申请的每个节点使用的核心数

# 每个节点的 gpu 数量
#SBATCH --gres=gpu:1                        ##作业申请的每个节点GPU卡数量

# 用户所在的用户组名称
#SBATCH --account=lh                         ##用户所在用户组名称

# 输出错误的文件名
#SBATCH --error=%j.err

# 运行完成的输出文件名字
#SBATCH --output=%j.out

nvidia-smi

CURDIR=`pwd`
rm -rf $CURDIR/nodelist.$SLURM_JOB_ID
NODES=`scontrol show hostnames $SLURM_JOB_NODELIST`
for i in $NODES
do
echo "$i:$SLURM_NTASKS_PER_NODE" >> $CURDIR/nodelist.$SLURM_JOB_ID
done


# 记录作业开始运行时间
echo "process will start at : "
date
echo "++++++++++++++++++++++++++++++++++++++++"

##以下几行为加载软件环境变量（注意：你在该任务里面需要用到的所有软件均需要添加到这个位置，务必根据实际情况按需添加或者删除）
##这部分根据实际情况修改，需要添加在作业脚本里面程序执行所需要用到的所有软件环境变量

#setting environment for your software           ##设置你在本作业需要用到的软件环境变量
#setting environment for inteloneapi2022.2  （这是一个案例）
#source  /public/software/intel/oneapi/2022.2/setvars.sh

# Program excute Command                             ##GPU程序执行命令语句，每个软件的执行命令都是互不相同的，请自行查询你所需要使用到的软件程序执行的命令格式，务必根据实际情况修改。软件跑串行、单节点多和并行、多节点多核并行、多线程请参考你所用到的软件的使用说明。
# 写你要执行的指令，比如启动训练，或者是加载conda
# module load apps/anaconda3/2021.05  # 加载anaconda
source activate czh_interactivate  # 加载anaconda虚拟环境

export CUDA_HOME=/public/home/lh/lh/zjn/environment/cuda-11.6/
export PATH="/public/home/lh/lh/zjn/environment/cuda-11.6/bin:$PATH"
export LD_LIBRARY_PATH="/public/home/lh/lh/zjn/environment/cuda-11.6/lib64:$LD_LIBRARY_PATH"

python /public/home/lh/lh/czh/interactivate/train_interactive.py --round 7 --model pointnet2_sem_seg --dataset powerline --lr_decay 1 --npoint 8192 --batch_size 32 --learning_rate 0.0005

# 记录作业运行结束时间
echo "++++++++++++++++++++++++++++++++++++++++"
echo "process will sleep 30s"
sleep 1
echo "process end at : "
date

# 删除 slurm 软件自动生成的临时文件
rm -rf $CURDIR/nodelist.$SLURM_JOB_ID
