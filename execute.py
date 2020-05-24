import gc, subprocess, os
import torch, torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as Image
from tqdm import tqdm 
from argparse import ArgumentParser
from models import SuperResolution

# Choose device for model running, the reconstruction is performed on the CPU by default
# This can be changed by simply replacing .cpu() with .to(device)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def execute(image_in, model, fs = 33, overlap = False, scale = 2):
    """
    Executes the model trained on colab, on any image given (link or local), with an 
    upscaling factor as mentioned in the arguments. For best results, use a scale of
    2 or lesser, since the model was trained on a scale of 2
    Inputs : image_in               -> torch.tensor representing the image, can be easily obtained from 
                                       transform_image function in this script (torch.tensor)
             model                  -> The trained model, trained using the same patch size 
                                       (object of the model class, inherited from nn.Module) 
             fs                     -> Patch size, on which the model is run (int)
             overlap                -> Reconstruction strategy, more details in the readme (bool)
             scale                  -> Scale on which the image is upscaled (float) 
    Outputs: reconstructed_image    -> The higher definition image as output (torch.tensor)
    """
    # Write the transforms and prepare the empty array for the image to be written
    c, h, w = image_in.shape
    scale_transform = transforms.Resize((int(h * scale), int(w * scale)), interpolation=3)

    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    image = to_tensor(scale_transform(to_pil(image_in)))
    n = 0
    c, h, w = image.shape
    image = image.unsqueeze(0)
    image = image.to(device)
    reconstructed_image = torch.zeros_like(image).cpu()
    reconstructed_image_weights = torch.zeros_like(image).cpu()

    # Loop for overlapping reconstruction 
    # Preferably avoid, needs too much memory even for small images 
    if overlap:
      for i in tqdm(range(h - fs + 1), desc = 'Progressively Scanning'):
        for j in range(w - fs + 1):
          gc.collect()
          patch = image[:, :, i: i + fs, j: j + fs]
          reconstructed_image[:, :, i: i + fs, j: j + fs] += model(patch)[0].cpu().clamp(0, 1)
          reconstructed_image_weights[:, :, i: i + fs, j: j + fs] += torch.ones(1, c, fs, fs)
      reconstructed_image /= reconstructed_image_weights
    
    # Loop for non overlapping image reconstruction 
    # A more detailed explanation of reconstruction methods is mentioned in the readme
    else:
      for i in tqdm(range(h // fs), desc = 'Progressively Scanning', ncols = 100):
        for j in range(w // fs):

          # Clean up memory and track iterations
          gc.collect()
          n += 1

          # Get the j'th (fs, fs) shaped patch of the (i * fs)'th row, 
          # Upscale this patch and write to the empty array at appropriate location  
          patch = image[:, :, i * fs: i * fs + fs, j * fs: j * fs + fs]
          reconstructed_image[:, :, i * fs: i * fs + fs, j * fs: j * fs + fs] = model(patch)[0].cpu().clamp(0, 1)
          reconstructed_image_weights[:, :, i * fs: i * fs + fs, j * fs: j * fs + fs] += torch.ones(1, c, fs, fs)
          
          # This leaves the right and bottom edge black, if the width and height are not divisible by fs
          # Those edge cases are dealt with here
          if j == w // fs - 1:
              patch = image[:, :, i * fs: i * fs + fs, w - fs: w]
              reconstructed_image[:, :, i * fs: i * fs + fs, w - fs: w] = model(patch)[0].cpu().clamp(0, 1)
          if i == h // fs - 1:
              patch = image[:, :, h - fs: h, j * fs: j * fs + fs]
              reconstructed_image[:, :, h - fs: h, j * fs: j * fs + fs] = model(patch)[0].cpu().clamp(0, 1)
          
      # Make the right bottom patch, since none of the edge cases have covered it
      patch = image[:, :, h - fs: h, w - fs: w]
      reconstructed_image[:, :, h - fs: h, w - fs: w] = model(patch)[0].cpu().clamp(0, 1)
    
    # Print output image shape for verification 
    print("Channels = {}, Image Shape = {} x {}".format(c, w, h))
    return reconstructed_image

def transform_image(path_to_image):
    """
    To simplify the transformation of an image
    Input : path_to_image     -> local path to image file
    Output: to_tensor(image)  -> image stored as tensor (torch.tensor) 
    """
    image = Image.open(path_to_image)
    to_tensor = transforms.ToTensor()
    return to_tensor(image)

if __name__ == '__main__':

    # Parse required command line arguments
    parser = ArgumentParser()
    parser.add_argument('--image', type = str)
    parser.add_argument('--scale', type = float,  default = 2)
    parser.add_argument('--path', type = str, default = 'results/image.png')
    parser.add_argument('--saved', type = str, default = 'saved/isr_best.pth')
    args = parser.parse_args()

    # If image link is given, then download and direct path variable to this image 
    if args.image[:4] == 'http':
        subprocess.check_output('wget -O ' + args.path + ' ' + args.image, shell = True)
        path_to_image = args.path
    else:
        path_to_image = args.image

    # Instantiate model and load state dict using .pth file 
    model = SuperResolution()
    if torch.cuda.is_available():
      model.load_state_dict(torch.load(args.saved))
    else:
      model.load_state_dict(torch.load(args.saved, map_location={'cuda:0': 'cpu'}))
    model.to(device)
    model.eval()

    # Run the progressive scan to increase resolution of the image 
    transformed = transform_image(path_to_image)
    reconstructed = execute(transformed, model, scale = args.scale)  
    to_pil = transforms.ToPILImage()

    # Save the image in the same directory as the source
    out_image = to_pil(reconstructed.squeeze())
    out_image.save(path_to_image.rsplit('.')[0] + str('_upscaled.') + path_to_image.rsplit('.')[1])
    print("Image written to {}".format(path_to_image.rsplit('.')[0] + str('_upscaled.') + path_to_image.rsplit('.')[1])) 
