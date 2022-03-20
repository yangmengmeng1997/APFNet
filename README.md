## Attribute-Based Progressive Fusion Network for RGBT Tracking<br>
## This project is created base on<br>
--MDNet: Real-Time Multi-Domain Convolutional Neural Network Tracker Created by Ilchae Jung, Jeany Son, Mooyeol Baek, and Bohyung Han
## Prerequisites<br>
<ol>
  <li> python>=3 </li>	
  <li> pytorch>=1.0 </li>	
  <li> some others library functions </li>	
</ol>
<br>
For more detailed packages, refer to [MDNet](https://github.com/hyeonseobnam/py-MDNet).<br> 

## Pretrained model for APFNet<br>
In our tracker, we use MDNet as our backbone and extend to multi-modal tracker.We use imagenet-vid.pth as our pretrain model.Then we use this with the training model in GTOT and RGBT234 models to pre-train our dual-stream MDNet_RGBT backbone network.And thus we get the GTOT.pth and RGBT234.pth.<br>

