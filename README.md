Install Dependencies for GPU (with anaconda):
```
conda create --name runcsp_env python=3.10
conda activate runcsp_env
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
```
Change "+cu113" to "+cpu" if you have no gpu

To extract the data, execute the following commands in your shell:
```
cd data
unzip K-Col-Graphs.zip
cd ..
```


To train the RUNCSP models used in [ANYCSP](https://arxiv.org/abs/2208.10227) run:
```
train_all_kcol.sh
```

