import torch, torch.nn as nn
import torchvision
from dataloader import load_loader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class SuperResolution(nn.Module):
    def __init__(self, sub_image: int = 33, spatial: list = [9, 1, 5], filter: list = [64, 32], num_channels: int = 3):
        super().__init__()
        self.layer_1 = nn.Conv2d(num_channels, filter[0], spatial[0], padding = 4)
        self.layer_2 = nn.Conv2d(filter[0], filter[1], spatial[1])
        self.layer_3 = nn.Conv2d(filter[1], num_channels, spatial[2], padding = 2)
        self.relu = nn.ReLU()

    def forward(self, image_batch):
        x = self.layer_1(image_batch)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.layer_3(x)
        x = self.relu(x)
        return x #+ image_batch # Heuristic since laptop cannot handle complete training

if __name__ == '__main__':
    low_res_loader, high_res_loader = load_loader()
    model = SuperResolution()
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), 1e-04)
    writer = SummaryWriter()
    n = 0

    for epoch in tqdm(range(100)):
        for low_res, high_res in zip(low_res_loader, high_res_loader):
            low_res_batch, high_res_batch = low_res[0], high_res[0]
            if torch.cuda.is_available():
                low_res_batch, high_res_batch = low_res_batch.cuda(), high_res_batch.cuda()
            reconstructed_batch = model(low_res_batch)
            loss_fn = nn.MSELoss()
            loss = loss_fn(high_res_batch, reconstructed_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            to_pil, resize, to_tensor = torchvision.transforms.ToPILImage(), torchvision.transforms.Resize((48 * 7, 144 * 7)), torchvision.transforms.ToTensor()
            image = to_pil(torch.cat((low_res_batch[0], high_res_batch[0], reconstructed_batch[0]), dim = 2).cpu())
            image = to_tensor(resize(image))
            image = image.clamp(0, 1)
            n += 1
            psnr = 10 * torch.log10(1 / loss)

            writer.add_scalar("MSE loss between the images", loss * (255 ** 2), n)
            writer.add_scalar("PSNR", psnr, n)
            writer.add_image("Low Resolution Image ------ High Resolution Image ----- Reconstructed Image", image, n, dataformats='CHW')