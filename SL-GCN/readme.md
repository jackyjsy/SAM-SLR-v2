# Sign Language Graph Convolutional Networks (SL-GCN)
## Data preparation
1. Download [AUTSL](http://chalearnlap.cvc.uab.es/dataset/40/description/), [CSL/SLR500](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html), and [WLASL2000](https://dxli94.github.io/WLASL/) dataset following their instructions.

2. Extract whole-body keypoints data following the instruction in ../data_process/wholepose

3. Run the following code to prepare the data for GCN.
        cd data_gen/
        python autsl_gendata.py
        python gen_bone_data.py
        python gen_motion.py

4. The preprocessed skeleton data for AUTSL, SLR500, and WLASL2000 datasets are provided [here](https://drive.google.com/drive/folders/1VUQsh_nf70slT4YsC-UzTCAZ3jB_uFKX?usp=sharing). Please be sure to follow their rules and agreements when using the preprocessed data.

## Pretrained models
Our pretrained models are provided [here](https://drive.google.com/drive/folders/1PYEZVstHXd3msTCye1wllULyPxny_tEc?usp=sharing).

## Usage
### Train WLASL:
```
python main.py --config config/WLASL/train/train_joint.yaml

python main.py --config config/WLASL/train/train_bone.yaml

python main.py --config config/WLASL/train/train_joint_motion.yaml

python main.py --config config/WLASL/train/train_bone_motion.yaml
```

### Test:
```
python main.py --config config/WLASL/test/test_joint.yaml

python main.py --config config/WLASL/test/test_bone.yaml

python main.py --config config/WLASL/test/test_joint_motion.yaml

python main.py --config config/WLASL/test/test_bone_motion.yaml
```

### Multi-stream ensemble:
1. Copy the results .pkl files from all streams (joint, bone, joint motion and bone motion) to ensemble/ and renamed them correctly.
2. Follow the instruction in ensemble/ to obtained the results of multi-stream ensemble.