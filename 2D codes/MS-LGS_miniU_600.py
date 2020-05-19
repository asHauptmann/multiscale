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



ops = []
op_adjs = []
fbps = []
sizes = []
proj_shapes = []
etas = []

for i in range(5):
    n = 32 * 2 ** i
    stride = size//n
    spc = odl.uniform_discr([-128, -128], [128, 128], [n, n],dtype='float32')
    g = odl.tomo.cone_beam_geometry(spc, src_radius=500, det_radius=500, num_angles = nAngles//stride)
    rt = odl.tomo.RayTransform(spc, g, impl='astra_cuda')
    fbp_scaled = odl.tomo.fbp_op(rt,filter_type='Hann',frequency_scaling=0.6)
    
    ops.append(OperatorAsModule(rt).to(device))
    op_adjs.append(OperatorAsModule(rt.adjoint).to(device))
    fbps.append(OperatorAsModule(fbp_scaled).to(device))
    sizes.append(n)
    proj_shapes.append(g.partition.shape)

    test_cur = nn.functional.interpolate(test_images, (n, n), mode='bilinear')
    normal = op_adjs[i](ops[i](test_cur))
    opnorm = torch.sqrt(torch.mean(normal ** 2)) / torch.sqrt(torch.mean(test_cur ** 2))
    etas.append(1 / opnorm)
    
    

fbp_op_mod = OperatorAsModule(fbp_op).to(device)
fbp_opCo_mod = fbps[0]

test_fbp = fbp_op_mod(test_data)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
       nn.Conv2d(in_channels, out_channels, 3, padding=1),
       nn.GroupNorm(num_groups=4, num_channels=out_channels),
       nn.ReLU(inplace=True),
       nn.Conv2d(out_channels, out_channels, 3, padding=1),
       nn.GroupNorm(num_groups=4, num_channels=out_channels),
       nn.ReLU(inplace=True) )

class Iteration(nn.Module):
    def __init__(self, op, op_adj,fbp):
        super().__init__()
        self.op = op
        self.op_adj = op_adj
        self.fbp = fbp

        
        self.dconv_down1 = double_conv(2, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)
        self.dconv_down5 = double_conv(256, 512)
        self.convMulti2   = double_conv(3, 32)
        self.convMulti3   = double_conv(3, 32)
        self.convMulti4   = double_conv(3, 32)
        self.dconv_down2Multi = double_conv(32+32, 64)
        self.dconv_down3Multi = double_conv(64+32, 128)
        self.dconv_down4Multi = double_conv(128+32, 256)
        
        self.maxpool = nn.MaxPool2d(2)
        self.xUp4  = nn.ConvTranspose2d(512,256,2,stride=2,padding=0)
        self.xUp3  = nn.ConvTranspose2d(256,128,2,stride=2,padding=0)
        self.xUp2  = nn.ConvTranspose2d(128,64,2,stride=2,padding=0)
        self.xUp1  = nn.ConvTranspose2d(64,32,2,stride=2,padding=0)
        
        self.dconv_up4 = double_conv(512, 256)
        self.dconv_up3 = double_conv(256, 128)
        self.dconv_up2 = double_conv(128, 64)
        self.dconv_up1 = double_conv(64, 32)
        self.conv_last = nn.Conv2d(32, 1, 1)
        
        self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.gradsize = nn.Parameter(torch.ones(1, 1, 1, 1))
        

        


    def forward(self, cur, curs,grads,fbps, y,it):
        # Set gradient of (1/2) ||A(x) - y||^2
        normal = self.op(cur) - y
        grad = self.op_adj(normal)  
        if it < 10:
            dx = torch.cat([cur, etas[it] * grad], dim=1)
            ''' residual block'''
            conv1 = self.dconv_down1(dx)
            dx = self.maxpool(conv1)
            conv2 = self.dconv_down2(dx)
            dx = self.xUp1(conv2)        
            
            dx = torch.cat([dx, conv1], dim=1)         
            dx = self.dconv_up1(dx)
            dx = self.conv_last(dx)
        else:
            dx = torch.cat([cur, etas[it] * grad], dim=1)
            ''' grad U-net'''
            
            conv1 = self.dconv_down1(dx)
            x = self.maxpool(conv1)
            old = torch.cat([curs[it-1], etas[it-1] * grads[it-1],fbps[it-1]], dim=1)            
            dxMulti = self.convMulti2(old)
            dx = torch.cat([x, dxMulti], dim=1)
            conv2 = self.dconv_down2Multi(dx)
            x = self.maxpool(conv2)
            old = torch.cat([curs[it-2], etas[it-2] * grads[it-2],fbps[it-2]], dim=1)            
            dxMulti = self.convMulti3(old)
            dx = torch.cat([x, dxMulti], dim=1)
            conv3 = self.dconv_down3Multi(dx)
            
            x = self.maxpool(conv3)
            
            old = torch.cat([curs[it-3], etas[it-3] * grads[it-3],fbps[it-3]], dim=1)            
            dxMulti = self.convMulti4(old)
            dx = torch.cat([x, dxMulti], dim=1)
            conv4 = self.dconv_down4Multi(dx)
            
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
            dx = self.conv_last(x)

            
        return cur  + self.stepsize * dx    
    
    
class IterativeNetwork(nn.Module):
    def __init__(self, ops, op_adjs, fbps, sizes, proj_shapes, init_op, loss):
        super().__init__()
        self.sizes = sizes
        self.proj_shapes = proj_shapes
        self.init_op=init_op
        self.loss = loss
        self.op = ops
        self.op_adj = op_adjs
        self.fbp = fbps
        
    
        assert len(ops) == len(op_adjs) == len(sizes)
        
        for i, (op, op_adj,fbp) in enumerate(zip(ops, op_adjs,fbps)):
            iteration = Iteration(op=op, op_adj=op_adj,fbp=fbp)
            setattr(self, 'iteration_{}'.format(i), iteration)

    def forward(self, y, true, it, writer=None):
        n = self.sizes[0]
        y_current = nn.functional.interpolate(y, self.proj_shapes[0], mode='area')
        current = self.init_op(y_current)
        curs = []
        grads = []
        fbpys = []
        for i in range(len(self.sizes)):
            n = self.sizes[i]
            proj_shape = self.proj_shapes[i]
            if i < len(self.sizes)-1:
                y_current = nn.functional.interpolate(y, proj_shape, mode='area')
            else:
                y_current = y
                
            if i < 10:
                current = nn.functional.interpolate(current, (n, n), mode='bilinear')
                 
                iteration = getattr(self, 'iteration_{}'.format(i))
                current = iteration(current,curs,grads,fbpys, y_current,i)
                
                
                curs.append(current)
                                
            else:
                
                
                current = nn.functional.interpolate(current, (n, n), mode='bilinear')
                iteration = getattr(self, 'iteration_{}'.format(i))
                current = iteration(current,curs,grads,fbpys, y_current,i)
            
            if writer:
                util.summary_image(writer, 'iteration_{}'.format(i), current, it)
        return current,  self.loss(current, true)
    
      
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


train_writer = tensorboardX.SummaryWriter(comment="/train")
test_writer = tensorboardX.SummaryWriter(comment="/test")


iter_net = IterativeNetwork(ops=ops, 
                            op_adjs=op_adjs,
                            fbps=fbps,
                            sizes=sizes,
                            proj_shapes=proj_shapes,
                            init_op = fbp_opCo_mod,
                            loss=nn.MSELoss()).to(device)

optimizer = optim.Adam(iter_net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, nIter)

for it in range(nIter):
    scheduler.step()
    iter_net.train()
    data, images = generate_data()
    images = torch.from_numpy(images).float().to(device)
    projs = torch.from_numpy(data).float().to(device)

    optimizer.zero_grad()
    output, loss = iter_net(projs, images, it,
                            writer=train_writer if it % 50 == 0 else None)
    loss.backward()
    optimizer.step()


    if it % 25 == 0:
        summaries(train_writer, output, output, images, loss, it, do_print=False)
        
        iter_net.eval()
        outputTest, lossTest = iter_net(test_data, test_images, it, writer=test_writer)
        summaries(test_writer, outputTest, test_fbp, test_images, lossTest, it, do_print=True)
        
        
'''Eval and save test data'''

if(finalEval):
    iter_net.eval()
    
    it = nIter
    outputTest, lossTest = iter_net(test_data, test_images, it, writer=test_writer)
    summaries(test_writer, outputTest, test_fbp, test_images, lossTest, it, do_print=True)
    
    
    basePathSave = 'results/MSLGS_miniU/'       
    torch.save(iter_net.state_dict(),basePathSave + 'state_dict.pth')
    
    images = np.load('mayo/L286_volume_' + str(nAngles) + '.npy')
    data = np.load('mayo/L286_data_' + str(nAngles) + '_new.npy')
    test_images = torch.from_numpy(images).float().to(device)
    test_data = torch.from_numpy(data).float().to(device)

    output = np.zeros_like(images)
    lossVec = np.zeros(210)
    for itEval in range(210):
        outputTest, lossTest = iter_net(torch.unsqueeze(test_data[itEval],dim=0), torch.unsqueeze(test_images[itEval],dim=0),nIter+itEval+1, writer=test_writer)
        output[itEval] = outputTest.cpu().detach().numpy()
        lossVec[itEval]  = lossTest.cpu().detach().numpy()
    
               
    np.save(basePathSave + 'outputImages',output)
    np.save(basePathSave + 'lossImages',lossVec)                