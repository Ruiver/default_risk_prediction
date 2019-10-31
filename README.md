环境 python3.x tensorflow0.14
### 训练
python main.py --mode=train
### 测试：
demo_model填写data_path_save下面生成好的地址如1540982873
python main.py --mode=train --demo_model=xxxx
运行完后生成result.txt文件。

预警模型有两种：
第一种直接运行bow_classify下的py文件
第二种是运行bow_classify/fc_te--st下的py文件