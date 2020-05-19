import numpy as np
#import matplotlib.pyplot as plt
import odl
from pyOperator import OperatorAsModule
import torch
from torch import nn
from torch import optim
import tensorboardX
import util
from mayo_util import FileLoader, DATA_FOLDER, DATA_FOLDER_TEST

np.random.seed(42);
torch.manual_seed(42);


device = 'cuda'
learning_rate = 1e-3
log_interval = 20
iter4Net = 5
size = 512
mu_water = 0.02
photons_per_pixel = 8000
batch_size = 4
nIter = 20000
nAngles = 600

finalEval = True

n_data = 1

space = odl.uniform_discr([-128, -128], [128, 128], [size, size],dtype='float32')
geometry = odl.tomo.cone_beam_geometry(space, src_radius=500, det_radius=500, num_angles = nAngles)
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda') 
fbp_op = odl.tomo.fbp_op(ray_trafo,filter_type='Hann',frequency_scaling=0.6)
file_loader = FileLoader(DATA_FOLDER, exclude='L286')
    
def generate_data(validation=False):
    """Generate a set of random data."""
    n_iter = 1 if validation else n_data

    y_arr = np.empty((n_iter, 1, ray_trafo.range.shape[0], ray_trafo.range.shape[1]), dtype='float32')
    x_true_arr = np.empty((n_iter,1, space.shape[0], space.shape[1]), dtype='float32')

    for i in range(n_iter):
        if validation:
            fi = DATA_FOLDER_TEST + 'L286_FD_3_1.CT.0002.0142.2015.12.22.18.22.49.651226.358224370.npy'
            data = np.load(fi)
            phantom = space.element(np.rot90(data, -1))
        else:
            fi = file_loader.next_file()
            data = np.load(fi)
            phantom = space.element(np.rot90(data, -1))

        phantom /= 1000.0  # convert go g/cm^3

        data = ray_trafo(phantom)
        data = np.exp(-data * mu_water)


        noisy_data = odl.phantom.poisson_noise(data * photons_per_pixel)
        noisy_data = np.maximum(noisy_data, 1) / photons_per_pixel
        log_noisy_data = np.log(noisy_data) * (-1 / mu_water)


        x_true_arr[i,0] = phantom
        y_arr[i,0] = log_noisy_data

    return y_arr, x_true_arr
    


# Generate validation data
data, images = generate_data(validation=True)


test_images = torch.from_numpy(images).float().to(device)
test_data = torch.from_numpy(data).float().to(device)

# Make pytorch Modules from ODL operators
fwd_op_mod = OperatorAsModule(ray_trafo).to(device)
fwd_op_adj_mod = OperatorAsModule(ray_trafo.adjoint).to(device)
fbp_op_mod = OperatorAsModule(fbp_op).to(device)

test_fbp = fbp_op_mod(test_data)

# Compute opnorm
normal = fwd_op_adj_mod(fwd_op_mod(torch.from_numpy(images).float()))
opnorm = torch.sqrt(torch.mean(normal ** 2)) / torch.sqrt(torch.mean(torch.from_numpy(images).float() ** 2))
eta = 1 / opnorm

def double_conv(in_channels, out_channels):
    return nn.Sequential(
       nn.Conv2d(in_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),       
       nn.ReLU(inplace=True),
       nn.Conv2d(out_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),
       nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
               
        self.dconv_down1 = double_conv(n_in, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        
        self.dconv_down5 = double_conv(512, 1024)
       
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.xUp4  = nn.ConvTranspose2d(1024,512,2,stride=2,padding=0)
        self.xUp3  = nn.ConvTranspose2d(512,256,2,stride=2,padding=0)
        self.xUp2  = nn.ConvTranspose2d(256,128,2,stride=2,padding=0)
        self.xUp1  = nn.ConvTranspose2d(128,64,2,stride=2,padding=0)
        
        self.dconv_up4 = double_conv(512 + 512, 512)
        self.dconv_up3 = double_conv(256 + 256, 256)
        self.dconv_up2 = double_conv(128 + 128, 128)
        self.dconv_up1 = double_conv(64 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_out, 1)
        
        self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        inp = x
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        
        x = self.maxpool(conv4)  
        x = self.dconv_down5(x)
        x = self.xUp4(x)        
        
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.xUp3(x)        
        
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.xUp2(x)        
        
        x = torch.cat([x, conv2], dim=1)      
        x = self.dconv_up2(x)
        x = self.xUp1(x)        
        
        x = torch.cat([x, conv1], dim=1)         
        x = self.dconv_up1(x)
        update = self.conv_last(x)

        return inp + self.stepsize * update
    
    
def summaries(writer, result, fbp, true, loss, it, do_print=False):
    residual = result - true
    squared_error = residual ** 2
    mse = torch.mean(squared_error)
    maxval = torch.max(true) - torch.min(true)
    psnr = 20 * torch.log10(maxval) - 10 * torch.log10(mse)
    
    
    relative = torch.mean((result - true) ** 2) / torch.mean((fbp - true) ** 2)

    
    if do_print:
        print(it, mse.item(), psnr.item(), relative.item())

    writer.add_scalar('loss', loss, it)
    writer.add_scalar('psnr', psnr, it)
    writer.add_scalar('relative', relative, it)

    util.summary_image(writer, 'result', result, it)
    util.summary_image(writer, 'true', true, it)
    util.summary_image(writer, 'fbp', fbp, it)
    util.summary_image(writer, 'squared_error', squared_error, it)
    util.summary_image(writer, 'residual', residual, it)
    util.summary_image(writer, 'diff', result - fbp, it)

train_writer = tensorboardX.SummaryWriter(comment="/train")
test_writer = tensorboardX.SummaryWriter(comment="/test")

unet = UNet(n_in=1, n_out=1).to(device)
loss_train = nn.MSELoss()
loss_test = nn.MSELoss()


optimizer = optim.Adam(unet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, nIter)


for it in range(nIter):
    scheduler.step()    

    data, images = generate_data()
    images = torch.from_numpy(images).float().to(device)
    projs = torch.from_numpy(data).float().to(device)
    fbpOut = fbp_op_mod(projs)
    
    unet.train()    
    optimizer.zero_grad()
    
    output = unet(fbpOut)
    
    loss = loss_train(output, images)
    loss.backward()
    optimizer.step()

  
    
    if it % 25 == 0:
        summaries(train_writer, output, fbpOut, images, loss, it, do_print=False)
        unet.eval()
        outputTest = unet(test_fbp)
        lossTest = loss_test(outputTest, test_images)
        summaries(test_writer, outputTest, test_fbp, test_images, lossTest, it, do_print=True)


'''Eval and save test data'''

if(finalEval):
            
    basePathSave = 'results/Unet/' 
    torch.save(unet.state_dict(),basePathSave + 'state_dict.pth')
    
    images = np.load('mayo/L286_volume_' + str(nAngles) + '.npy')
    data = np.load('mayo/L286_data_' + str(nAngles) + '_new.npy')
    test_images = torch.from_numpy(images).float().to(device)
    test_data = torch.from_numpy(data).float().to(device)
    unet.eval()
    
    
    output = np.zeros_like(images)
    lossVec = np.zeros(210)
    for itEval in range(210):
        fbpOut = fbp_op_mod(torch.unsqueeze(test_data[itEval],dim=0))
        outputTest = unet(fbpOut)
        lossTest = loss_test(outputTest, torch.unsqueeze(test_images[itEval],dim=0))
        output[itEval] = outputTest.cpu().detach().numpy()
        lossVec[itEval]  = lossTest.cpu().detach().numpy()
    
               
    np.save(basePathSave + 'outputImages',output)
    np.save(basePathSave + 'lossImages',lossVec)
    
            