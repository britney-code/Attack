import os 
import tqdm 
import torch 
import argparse 
import sys 
from PIL import Image
dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(dir)
sys.path.append(project_root)
from transferattack.utils import *
from torchvision import transforms
from io import BytesIO
import torch.nn.functional as F
from scipy.fftpack import dct, idct
import numpy as np
_to_pil_image = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()


def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-n','--name', default="fd",type=str, help='defense method')
    parser.add_argument('-e', '--eval',default=True, action='store_true', help='attack/evluation')
    parser.add_argument('--batchsize', default=20, type=int, help='the bacth size')
    parser.add_argument('--model', default='inceptionv3', type=str, help='the source surrogate model')
    parser.add_argument('--input_dir', default='./data/', type=str,help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--adv_dir', default='./result/mifgsm', type=str, help='the path to store the adversarial patches')
    parser.add_argument("--process_adv_dir", default="./defense/result/", type=str, help = "process adversarial example path")
    parser.add_argument('--targeted',default=False, action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='0', type=str)
    return parser.parse_args()


class Randomization(object):
    '''Random input transform defense method.'''
    def __init__(self, device='cuda', prob=0.8, crop_lst=[0.1, 0.08, 0.06, 0.04, 0.02]):
        '''
        Args:
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            prob (float): The probability of input transform.
            crop_lst (list): The list of the params of crop method.
        '''
        self.prob = prob
        self.crop_lst = crop_lst
        self.device = device

    def __call__(self, images):
        '''The function to perform random transform on the input images.'''
        images = self.input_transform(images)
        return images

    def input_transform(self, xs):
        p = torch.rand(1).item()
        if p <= self.prob:
            out = self.random_resize_pad(xs)
            return out
        else:
            return xs

    def random_resize_pad(self, xs):
        rand_cur = torch.randint(low=0, high=len(self.crop_lst), size=(1,)).item()
        crop_size = 1 - self.crop_lst[rand_cur]
        pad_left = torch.randint(low=0, high=3, size=(1,)).item() / 2
        pad_top = torch.randint(low=0, high=3, size=(1,)).item() / 2
        if len(xs.shape) == 4:
            bs, c, w, h = xs.shape
        elif len(xs.shape) == 5:
            bs, fs, c, w, h = xs.shape
        w_, h_ = int(crop_size * w), int(crop_size * h)
        # out = resize(xs, size=(w_, h_))
        out = F.interpolate(xs, size=[w_, h_], mode='bicubic', align_corners=False)
        pad_left = int(pad_left * (w - w_))
        pad_top = int(pad_top * (h - h_))
        out = F.pad(out, [pad_left, w - pad_left - w_, pad_top, h - pad_top - h_], value=0)
        return out

class BitDepthReduction(object):
    '''Bit depth reduction defense method.'''
    def __init__(self, device='cuda', compressed_bit=4):
        '''
        Args:
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            compressed_bit (int): The compressed bit.
        '''
        self.compressed_bit = compressed_bit
        self.device = device
    
    def __call__(self, images):
        '''The function to perform bit depth reduction on the input images.'''
        images = self.bit_depth_reduction(images)
        return images

    def bit_depth_reduction(self, xs):
        bits = 2 ** self.compressed_bit    #2**i
        xs_compress = (xs.detach() * bits).int()
        xs_255 = (xs_compress * (255 / bits))
        xs_compress = (xs_255 / 255).to(self.device)
        return xs_compress
    
class Jpeg_compression(object):
    '''JPEG compression defense method.'''
    def __init__(self, device='cuda', quality=70):
        '''
        Args:
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            quality (int): The compressed image quality.
        '''
        self.quality = quality
        self.device = device
    
    def __call__(self, images):
        '''The function to perform JPEG compression on the input images.'''
        images = self.jpegcompression(images) 
        return images

    def jpegcompression(self, x):
        lst_img = []
        for img in x:
            img = _to_pil_image(img.detach().clone().cpu())
            virtualpath = BytesIO()
            img.save(virtualpath, 'JPEG', quality=self.quality)
            lst_img.append(_to_tensor(Image.open(virtualpath)))
        return x.new_tensor(torch.stack(lst_img))
    
class FD: 
    def __init__(self, num = 8): 
       self.num = num 
       self.q_table = np.ones((self.num, self.num)) * 30
       self.q_table[0:4, 0:4] = 25
    
    def dct2(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')   
    
    def idct2(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def __call__(self, input_matrix):
        input_matrix = input_matrix.cpu().numpy()
        input_matrix = input_matrix.transpose(0, 2, 3, 1)
        output = []
        input_matrix = input_matrix * 255
        n = input_matrix.shape[0]
        # TODO:  224的图片不需要这一步
        input_matrix = np.array([np.array(Image.fromarray(np.uint8(input_matrix[i])).resize((304, 304))) for i in range(n)])
        h = input_matrix.shape[1]
        w = input_matrix.shape[2]
        c = input_matrix.shape[3]
        
        horizontal_blocks_num = w / self.num
        output2 = np.zeros((c, h, w))
        output3 = np.zeros((n, 3, h, w))
        vertical_blocks_num = h / self.num
        n_block = np.split(input_matrix, n, axis=0)
        for i in range(1, n):
            c_block = np.split(n_block[i], c, axis=3)
            j = 0
            for ch_block in c_block:
                vertical_blocks = np.split(ch_block, vertical_blocks_num, axis=1)
                k = 0
                for block_ver in vertical_blocks:
                    hor_blocks = np.split(block_ver, horizontal_blocks_num, axis=2)
                    m = 0
                    for block in hor_blocks:
                        block = np.reshape(block, (self.num, self.num))
                        block = self.dct2(block)
                        # quantization
                        table_quantized = np.matrix.round(np.divide(block, self.q_table))
                        table_quantized = np.squeeze(np.asarray(table_quantized))
                        # de-quantization
                        table_unquantized = table_quantized * self.q_table
                        IDCT_table = self.idct2(table_unquantized)
                        if m == 0:
                            output = IDCT_table
                        else:
                            output = np.concatenate((output, IDCT_table), axis=1)
                        m = m + 1
                    if k == 0:
                        output1 = output
                    else:
                        output1 = np.concatenate((output1, output), axis=0)
                    k = k + 1
                output2[j] = output1
                j = j + 1
            output3[i] = output2

        output3 = np.transpose(output3, (0, 2, 1, 3))
        output3 = np.transpose(output3, (0, 1, 3, 2))
        # TODO: 224 不需要这一步
        output3 = np.array([np.array(Image.fromarray(np.uint8(output3[i])).resize((299, 299))) for i in range(n)])
        output3 = output3 / 255
        output3 = np.clip(np.float32(output3), 0.0, 1.0)
        return torch.from_numpy(output3.transpose(0, 3, 1, 2)).cuda()

def main():
    args = get_parser()
    process_adv_dir = os.path.join(args.process_adv_dir, args.name)
    if not os.path.exists(process_adv_dir):
        os.makedirs(process_adv_dir)
    bit_red = BitDepthReduction()
    jpeg = Jpeg_compression()
    rp = Randomization()
    fd = FD()
    
    print(f"===> 正在执行{args.name}预处理方法! ==> 保存的路径为:{process_adv_dir} ")
    dataset = AdvImagetNet(input_dir=args.input_dir, output_dir=args.adv_dir, targeted=args.targeted, eval=args.eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)
    for batch_idx, [images, labels, filenames] in enumerate(tqdm.tqdm(dataloader)):
        process_images = fd(images)
        save_images(process_adv_dir, process_images, filenames)

    
if __name__ == '__main__':
    main()