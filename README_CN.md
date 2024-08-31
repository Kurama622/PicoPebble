**[ENGLISH](./README.md)**  |  **中文版**

# 介绍

PicoPebble是一个面向机器学习领域初学者的分布式机器学习训练框架。它通过mpi在多机之间传递参数以及更新梯度，当然你也可以只用单机训练。目前PicoPebble已经支持的功能包括：
- 同步训练
- 异步训练
- 数据并行
- pipeline模型并行

还有一些功能也在开发计划内：
- tensor模型并行
- 通过gloo传递参数
- 灾备

# 依赖

目前PicoPebble依赖mpi来做参数同步，因此您需要安装openmpi，请注意不要同时安装openmpi和mpich

## Docker

```bash
docker build -t picopebble -f Dockerfile .

# 如果你使用podman
# podman build -t picopebble -f Dockerfile .`
```

## Ubuntu
```bash
sudo apt install openmpi-bin libopenmpi-dev
```

## Archlinux

```bash
sudo pacman -S openmpi
```


# 编译运行

## 单机单节点
```bash
# ./build_run.sh <node num>
./build_run.sh 1
```

## 多节点
```bash
./build_run.sh 3
```

# 参考

- [https://foundationsofdl.com/2022/02/12/neural-network-from-scratch-part-5-c-deep-learning-framework-implementation/](https://foundationsofdl.com/2022/02/12/neural-network-from-scratch-part-5-c-deep-learning-framework-implementation/)
