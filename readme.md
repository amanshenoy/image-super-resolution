## Image Super Resolution - Current Status 

The following the repository for NNFL project based on the paper 'Single Image Super Resolution using Deep ConvNets"   
  
The code in the current commit is **just a basic model and will be added to strongly**. The current Naive implementation achieves a PSNR of 30 (Paper achieves ~33). Data has been collected very naively and is a part of the further work. `data` folder has been added to .gitignore since the size is too large for uploading.  
  
The below tasks will be divided amongst all members and commits will be only made for significant changes.  
  
Everything has been logged using tensorboard because it is interactive and pretty. For every update the MSE Loss and PSNR are plotted scalars, and an image array of 'Low res | High res | Reconstructed' is also added to tensorboard.  
  
## Set of tasks to complete next
  
* Reconstruction function to be able to get the whole image from all the reconstructed patches.    
* Extensive experiments with hyperaparameters and variants of the network (are also hyperparameters, but experimented on independantly)  
* Current dataloader applies a simple transform CentreCrop instead of every patch in the image. This has to be changed to training on all patches in an image.  

## Potential set of tasks - Secondary priority
  
* Try on a new dataset?  
* Play with network configurations not mentioned in the paper  
* Visualize intermediate layers to interpret what the model is doing  