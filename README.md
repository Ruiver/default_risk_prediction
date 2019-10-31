# Requirment python3.x tensorflow0.14
# FOR EBCC
### Training
python main.py --mode=train
note that because of the size limit of upload file, in this version, we didnt upload the ELMo embedding, readers can refer to https://github.com/berkay-onder/ELMoForManyLangs
### testing：
after training the EBCC model saved in data_path_save directory with the file name in the form of timestamp like 1540982873.
Than run
python main.py --mode=all --demo_model=file  name, notice that all_2 parameter mean apply model to both reviews and posts, in this release version we only uses posts. Please contact corresponding author to get dataset. https://pan.baidu.com/s/1N_m_6hTqtSjbLKyIwpEkFQ. The keywords extraction result will be generated in result.txt.

# For platform default risk prediction
### After result.txt文件。
run bow_classfy/*py
or run bow_classify/fc_test/*.py

# Once the paper on public, all the datasets and pretrained model will release in this address.
