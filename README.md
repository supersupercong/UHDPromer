# UHDPromer [IJCV-26]

This is the official PyTorch codes for the paper  
[Neural Discrimination-Prompted Transformers for Efficient UHD Image Restoration and Enhancement](https://link.springer.com/article/10.1007/s11263-025-02662-6)  
[Cong Wang\*](https://supercong94.wixsite.com/supercong94), Jinshan Pan, Liyan Wang, Wei Wang\*, and Yang Yang    
<!--(\* indicates equal contribution) -->

![framework_img](framework_overview.png)

<!--
Here are some example results on test images from [BSRGAN](https://github.com/cszn/BSRGAN) and [RealESRGAN](https://github.com/xinntao/Real-ESRGAN).
-->
---

## Dependencies and Installation

- Ubuntu >= 18.04
- CUDA >= 11.0
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/supersupercong/UHDPromer.git
cd FeMaSR 

# create new anaconda env
conda create -n femasr python=3.8
source activate femasr 

# install python dependencies
pip3 install -r requirements.txt
python setup.py develop
```

## Train the model


### Dataset

[UHD-LL](https://drive.google.com/drive/folders/1IneTwBsSiSSVXGoXQ9_hE1cO2d4Fd4DN), [UHD-Haze](https://drive.google.com/drive/folders/1PVCPkhqU_voPVFZj3FzAtUkJnQnF9lSa), [UHD-Blur](https://drive.google.com/drive/folders/1O6JYkOELLhpEkirAnxUB2JGWMqgwVvmX)

### Train model

```
bash train.sh
```

### Test Model


```
bash test.sh
```

### Visual results
[UHD-LL&UHD-Haze&UHD-Blur&LOL-GoPro-SOTS]()


## Citation
```
@article{2026Neural,
  title={Neural Discrimination-Prompted Transformers for Efficient UHD Image Restoration and Enhancement},
  author={ Wang, Cong  and  Pan, Jinshan  and  Wang, Liyan  and  Wang, Wei  and  Yang, Yang },
  journal={International Journal of Computer Vision},
  volume={134},
  number={3},
  year={2026},
}
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Acknowledgement

This project is based on [BasicSR](https://github.com/xinntao/BasicSR).
