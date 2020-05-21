# Image Super Resolution using Deep Convolutional Networks - PyTorch Implementation
The following is the repository for project component of the course Neural Networks and Fuzzy Logic by - Aman Shenoy, Arnav Gupta, and Nikhil Gupta. The rest of this readme will follow the presentation structure (to maintain consistency), with instructions on how to run the code towards the end. Original paper can be found [here](https://arxiv.org/abs/1501.00092) 

Our models have been trained with a scaling factor of 2, so the image during model exectution can be upscaled to upto twice its original size. (One can choose the scale at which the image is to be upscaled during execution).  

## Datasets Used and Augmentation Schemes
Our original intention was to use ImageNet for both training and testing, but since it is no more available on PyTorch's dataset module, we have instead used [STL-10](https://ai.stanford.edu/~acoates/stl10/) for training, and images from [TextVQA dataset](https://textvqa.org/dataset) (Already had it sitting in drive) for testing. 

As mentioned, we were to use any two augmentation schemes in our implementation. We used the torchvisions transform TenCrop to be able to do within torchvision transforms. For every image, TenCrop generated 10 different augments (4 corner crops + 1 center crop - all of these horizontally flipped), thus increasing our dataset size ten-fold. Since the transform gives a tuple and not a torch tensor, this problem had to be specifically handled during training.

## Quantitative Results
Shown below are the results of our model, during the progress of training on test and training set.  

The blue curve represents the testing metrics, and the orange are the training set metrics.  

| Peak Signal to Noise ratio for 10k iterations |
:---------------------------------------------|
|![PSNR](https://github.com/amanshenoy/image-super-resolution/blob/master/demonstrations/PSNR.svg)|
The above curve has been plotted to be compared with table 1 in the paper. We used the model with channel depths of 128 and 64, equivalent to the left most model in table 1 of the paper. We have used a filter configuration of 9-5-5 for successive layers.  

The paper achieves a PSNR of 32.60 in 0.6 seconds. In comparision we achieve a test PSNR of 28.00 in 0.05 seconds (and 29.28 for training PSNR). We timed the testing using tqdm. 
  
Our implementation achieves lesser PSNR mainly due to lesser iterations trained (They trained for 10^8 iterations, whereas we trained for 10^4 iterations). Our implementation is significantly faster because of the extensive use of CUDA wherever possible.

| Mean Squared Error between image and reconstruction from the downscaled |
:-------------------------------------------------------------------------:
|![MSE](https://github.com/amanshenoy/image-super-resolution/blob/master/demonstrations/MSE%20loss%20between%20the%20images.svg)|
Even though PSNR and MSE can be related mathematically, MSE has been plotted since it is the loss function we have minimized during training

| The average test PSNR of BiCubic interpolation (like the paper) on the images | 
:-----------------------------------------------------------------------------:
|![bicubic](https://github.com/amanshenoy/image-super-resolution/blob/master/demonstrations/PSNR%20of%20BiCubic%20Interpolation%20(For%20comparision)%20(1).svg)|
This is important because, uptil this value the model had not learned anything useful. PSNR beyond this value is when the model begins to learn something.  

## Qualitative Results
**We have focused extensively on qualitative results due to the nature of the project. All visualizations and animations were made with extensive use of tensorboard and ffmpeg.** Below is an example of the result from the execute function -
| Original Image | Reconstructed |
:---------------:|:--------------:|
![](https://github.com/amanshenoy/image-super-resolution/blob/master/results/monarch.bmp) | ![](https://github.com/amanshenoy/image-super-resolution/blob/master/results/monarch_upscaled.bmp)

Along with the output, we have also generated animations that help better interpret what happens in the intermediate layers.   

Below shows two images along with their reconstructions as training progresses for the first 20 epochs. The generated `.gif` runs for 4 seconds and is generated at 5fps (evaluating to 20 frames, 1 for every epoch of the first 20)
| Image patch 1 | Training Progress | Image patch 2 | Training progress |
:--------------:|:-----------------:|:--------------:|:-----------------:|
![](https://github.com/amanshenoy/image-super-resolution/blob/master/demonstrations/patch_25.png) | ![](https://github.com/amanshenoy/image-super-resolution/blob/master/demonstrations/r1_other.gif) | ![](https://github.com/amanshenoy/image-super-resolution/blob/master/demonstrations/patch_30.png) | ![](https://github.com/amanshenoy/image-super-resolution/blob/master/demonstrations/r2_other.gif) 

Along with this, we have also visualized intermediate layers for interpretation. For the volume generated after the second convolutional layer, we visualized random channels (out of the 64 channels of the output of that layer), for the same 2 inputs above.

| Channel 16/64 on image patch 1 | Channel 00/64 on image patch 1| Channel 05/64 on image patch 2 | Channel 08/64 on image patch 2 |
:--------------:|:-----------------:|:--------------:|:-----------------:|
![](https://github.com/amanshenoy/image-super-resolution/blob/master/demonstrations/channel_16.gif) | ![](https://github.com/amanshenoy/image-super-resolution/blob/master/demonstrations/channel_0.gif) | ![](https://github.com/amanshenoy/image-super-resolution/blob/master/demonstrations/channel_5.gif) | ![](https://github.com/amanshenoy/image-super-resolution/blob/master/demonstrations/channel_8.gif) 

This lets us interpret what the kernels of the convolutional layers try to learn. As we can see from the animations above, some kernels seem to segment subject and background (Channel 08/64 on image patch 2) and some kernels act as edge detectors (Channel 16/64 on image patch 1).

Once this model is trained, we use two strategies to execute - overlapping and non-overlapping. 

Overlapping would evaluate the model on every possible 33 x 33 patch of the image, and the value for a pixel is the average of all the values obtained for that pixel (each pixel can be a part of multiple patches). e.g. The first patch diagonal is (0, 0) to (33, 33) ; second patch diagonal is (1, 0) to (34, 33)

Non-overlapping on the other hand evlautes the model on each pixel only once e.g. The first patch diagonal is (0, 0) to (33, 33) ; second patch diagonal is (33, 0) to (66, 33).   

Since overlapping evaluates much lesser patches, it is much quicker and uses significantly lesser memory. The progressive (top to bottom) run over an image is visualized below. 

|Top to bottom progressive scan using non-overlapping patches (Look carefully to be able to spot the scan)|
|:----------------------------------------------------------:|
|![](https://github.com/amanshenoy/image-super-resolution/blob/master/demonstrations/progressive.gif)|

## How to Implement
Implementation begins with the only assumption that `pip` and `python` are installed and functional. 

    > git clone https://github.com/amanshenoy/image-super-resolution.git
    > cd image-super-resolution
    > pip install -r requirements.txt

This is all that is required to set the repository up. 

To be able to retrain the model, instead of running locally, its better to run on google colab. This can be done by opening the notebook `training.ipynb` on GitHub and clicking on the 'Open in Colab' badge. Once this is done running the first cell will prompt for an authorization code for accessing your google drive (for regular saving) for which the instructions as given in the cell are to be followed. The training process is constantly visualized on tensorboard for interpretation, and models are constantly saved to drive. 

Evaluation or execution of the model can be done without running training, since we have saved our models in `saved` folder.

To evaluate the model on an image link given by - `https://raw.githubusercontent.com/amanshenoy/image-super-resolution/master/results/barbara.bmp`

    > python execute.py --image https://raw.githubusercontent.com/amanshenoy/image-super-resolution/master/results/barbara.bmp --scale 2 --path results/download.png --saved saved/isr_best.pth

Where `--image` - parses an image link, `--scale` - is the scale to upscale by (Less then 2 reccommended, since model was trained with scale 2), `--path` - is the path to store the image given by the link (Output will be saved in the same directory), and `--saved` - is the path to the trained model used

To evaluate the model on a local image saved at `results/flower.bmp`

    > python execute.py --image results/flower.bmp --scale 2 --saved saved/isr_best.pth

Where all the flags are the same as above. The `--path` flag for a local image is meaningless and is never used.

**All random initializations have used the defualt torch settings, where the random weights for a layer are sampled from a uniform distribution (-1/root(m), 1/root(m)), where m is fan in for that layer**
