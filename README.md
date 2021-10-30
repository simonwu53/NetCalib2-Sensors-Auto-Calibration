# This is The Way: Sensors Auto-calibration Approach Based on Deep Learning for Self-driving Cars

This work is accepted for publication in IEEE Sensors Journal.

<img src="assets/pipeline.png" width="1600" alt="pipeline-figure">

In addition to the original deep neural network architecture, we added different losses to constrain the convergence of the model. Also, the preprocessing of dataset is required for fast training and convergence.

## Motivation
We are releasing our approach's source code for auto-calibrating camera and lidar sensors to share with the scientific community and industry with the aim of the collaboration with people around the world to push the boundaries in the field of autonomous driving and intelligent transportation systems. 

## Experiment results
### 1. Calibration Errors

The figure shows the comparison between decalibration (x-axis, artificial errors) and errors after auto-calibration (y-axis) for rotational and translational axes. 

<img src="assets/calib_err.png" width="1000" alt="calibration-errors">

### 2. Visualization

<img src="assets/eval.png" width="1600" alt="eval">

<img src="assets/eval_zoomed.png" width="400" alt="eval-zoomed">

### 3. Speed
| Method | second per frame |
| ---  | --- |
| Ours | 0.0047 |
| [Go-ICP](https://github.com/yangjiaolong/Go-ICP) | 1.8221 |
| [RGGNet](https://github.com/KleinYuan/RGGNet) | 0.0244 |

## Deployment
### 1. System Environment

The method is tested based on the following setup, it may run on other setup as well. 
Version of packages may not necessarily fully match the numbers below.

| Env | Version |
| ---  | --- |
| System | 20.04 |
| Python | 3.8 |
| Numpy | 1.18.4 |
| PyTorch | 1.5.0 |
| torchvision | 0.6.0 |
| tqdm | 4.46.0 |
| Pillow | 7.1.2 |

For a full list of all dependencies, please visit [here](https://github.com/simonwu53/NetCalib2-Sensors-Auto-Calibration/blob/master/requirements.txt).

1. Clone the repo to your local drive. 
2. Setup your own Python environment and install necessary dependencies.
3. The installation of `pytorch` and `torchvision` may vary in the future because of variable CUDA versions, you can install manually via the [official website](https://pytorch.org/get-started/locally/).

### 2. Train the model
Modify the configurations in `src/train.sh` and read the available parameters from `src/train.py`. 
Run the training process is as simple as run the following command after configured the parameters.

```shell script
bash train.sh
```

In order to continue training, you should modify the configuration in `train.sh` to use the checkpoint file you have saved.

## Test the model
You can test the trained model by the command below,

```shell script
python eval.py --ckpt ../results/HOPE2/ckpt/Epoch47_val_0.0475.tar --visualization --rotation_offsest 10 --translation_offsest 0.2
```

* Use `--visualization` flag to show the video feed window (OpenCV required).
* `--rotation_offsest` and `--translation_offsest` used during the training process are required to correctly recognize the output from the model.

## Licence 
NetCalib is released under a [MIT License](https://github.com/simonwu53/NetCalib2-Sensors-Auto-Calibration/blob/master/LICENSE) license. 

For a closed-source version of NetCalib for commercial purposes, please contact the authors: [Wu](mailto:Shan.Wu@ut.ee) and [Hadachi](mailto:hadachi@ut.ee)


## Contributors
Shan Wu; Amnir Hadachi; Damien Vivet; Yadu Prabhakar.  

## Citation 
If you use NetCalib in an academic work, please cite:
```
@inproceedings{Shan2020NetCalib,
  title={NetCalib: A Novel Approach for LiDAR-Camera Auto-calibration Based on Deep Learning},
  author={Shan, Wu; Hadachi, Amnir; Vivet, Damien; Prabhakar, Yadu},
  booktitle={Proceedings of the 25th International Conference on Pattern recognition 2020},
  year={2020},
  organization={IEEE}
}
```

Preprint version of the paper is [here](https://www.researchgate.net/publication/344694742_NetCalib_A_Novel_Approach_for_LiDAR-Camera_Auto-calibration_Based_on_Deep_Learning).


