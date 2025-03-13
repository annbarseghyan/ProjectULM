# UltraSound Localization Microscopy (ULM) Project  

UltraSound Localization Microscopy (ULM) is a technique for high-resolution blood flow imaging. It achieves this by tracking microbubbles in contrast-enhanced ultrasound images. This project provides implementations of ULM for microbubble localization and tracking in ultrasound data, featuring two approaches:  

- **Classical ULM**: Based on localization methods described in the PALA paper [1].  
- **Deep Learning ULM**: Utilizes a neural network model inspired by the DeepLoco paper for localization [2].  

## Data & Pretrained Model  
**Trained Model & Generated Data** are available at:  
[https://drive.google.com/file/d/1P-nifg2JLSd4dZYWeIzQCW47dsL8_RzO/view?usp=sharing]  

Download the files and place them in the appropriate directories before running the scripts.  

## Running the ULM Pipeline  

### **DeepLoco-Based ULM**  
To perform localization and tracking using the DeepLoco-inspired deep learning approach, run:  
```bash
python localize_and_track.py
```

### **Classical ULM**  
For classical ULM tracking described in pala paper, run:
```sh
python localize_and_track_pala.py
```

## References  
This project builds on the following research papers:  

1. B. Heiles et al., "Performance benchmarking of microbubble-localization algorithms for ultrasound localization microscopy," *Nature Biomedical Engineering*, vol. 6, no. 5, pp. 605–616, May 2022.  

2. N. Boyd, E. Jonas, H. Babcock, and B. Recht, "DeepLoco: Fast 3D Localization Microscopy Using Neural Networks," *bioRxiv* preprint, Feb. 16, 2018.  

