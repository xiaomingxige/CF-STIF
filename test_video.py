import torch
import numpy as np
from collections import OrderedDict
from Network import Net
import utils
from tqdm import tqdm
import glob


def get_h_w_f(filename):
    WxH = filename.split('_')[-2]   
    frame_nums = int((filename.split('_')[-1]).split('.')[0])    
    width = int(WxH.split('x')[0])       
    height = int(WxH.split('x')[1])        
    return height, width, frame_nums


from PIL import Image
from numpy import *
def yuv2rgb(Y, U, V):
    height, width = (Y.shape)
    U = np.array(Image.fromarray(U).resize((width, height)))
    V = np.array(Image.fromarray(V).resize((width, height)))
    rf = Y + 1.4075 * (V - 128.0)
    gf = Y - 0.3455 * (U - 128.0) - 0.7169 * (V - 128.0)
    bf = Y + 1.7790 * (U - 128.0)
    for m in range(height):
        for n in range(width):
            if(rf[m, n] > 255):
                rf[m, n] = 255
            if(gf[m, n] > 255):
                gf[m, n] = 255
            if(bf[m, n] > 255):
                bf[m, n] = 255
            if (rf[m, n] < 0):
                rf[m, n] = 0
            if (gf[m, n] < 0):
                gf[m, n] = 0
            if (bf[m, n] < 0):
                bf[m, n] = 0
    r = rf.astype(uint8)
    g = gf.astype(uint8)
    b = bf.astype(uint8)
    return r, g, b


ckp_path = './exp/QP37/ckp_255000.pt'  # trained at QP37, LDP, HM16.5


def test_one_video(lq_yuv_path, raw_yuv_path):
    h, w, nfs = get_h_w_f(raw_yuv_path)
    # ==========
    # Load pre-trained model
    # ==========
    opts_dict = {'radius': 3, 
        'mlrd': {'in_nc': 1, 'm_nc': 64, 'out_nc': 64, 'bks': 3, 'dks': 3, }, 
        'qe': {'in_nc': 64, 'm_nc': 64, 'out_nc': 1, 'bks': 3, },}   

    model = Net(opts_dict=opts_dict)
    checkpoint = torch.load(ckp_path)

    torch.save(checkpoint, "./ckp_255000.pt", _use_new_zipfile_serialization=False)  

    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()
    # ==========
    # Load entire video
    # ==========
    raw_y = utils.import_yuv(seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True) 
    lq_y = utils.import_yuv(seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True)
    raw_y = raw_y.astype(np.float32) / 255.
    lq_y = lq_y.astype(np.float32) / 255.
    # ==========
    # Define criterion
    # ==========
    criterion_psnr = utils.PSNR()
    unit = 'dB'
    criterion_ssim = utils.SSIM()
    # ==========
    # Test
    # ==========
    pbar = tqdm(total=nfs, ncols=80)
    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()

    ori_ssim_counter = utils.Counter()
    enh_ssim_counter = utils.Counter()
    with torch.no_grad():  
        for idx in range(opts_dict['radius'], nfs - opts_dict['radius']):   
            idx_list = list(range(idx-3, idx+4))  
            idx_list = np.clip(idx_list, 0, nfs-1)  
            input_data = []
            for idx_ in idx_list:
                input_data.append(lq_y[idx_])
            input_data = torch.from_numpy(np.array(input_data))
            input_data = torch.unsqueeze(input_data, 0).cuda()  
            # enhance
            enhanced_frm = model(input_data)

            # eval
            gt_frm = torch.from_numpy(raw_y[idx]).cuda()
            batch_pre_psnr = criterion_psnr(input_data[0, 3, ...], gt_frm)
            batch_aft_psnr = criterion_psnr(enhanced_frm[0, 0, ...], gt_frm)
            ori_psnr_counter.accum(volume=batch_pre_psnr)
            enh_psnr_counter.accum(volume=batch_aft_psnr)

            batch_pre_ssim = criterion_ssim(input_data[0, 3, ...], gt_frm)
            batch_aft_ssim = criterion_ssim(enhanced_frm[0, 0, ...], gt_frm)
            ori_ssim_counter.accum(volume=batch_pre_ssim)
            enh_ssim_counter.accum(volume=batch_aft_ssim)


            # display
            pbar.set_description("[{:.3f}]{:s} -> [{:.3f}]{:s}".format(batch_pre_psnr, unit, batch_aft_psnr, unit))
            pbar.update()
    pbar.close()
    ori_psnr = ori_psnr_counter.get_ave()
    enh_psnr = enh_psnr_counter.get_ave()

    ori_ssim = ori_ssim_counter.get_ave()
    enh_ssim = enh_ssim_counter.get_ave()
    print('PSNR:  ave ori[{:.3f}]{:s}, enh[{:.3f}]{:s}, delta[{:.3f}]{:s}'.format(ori_psnr, unit, enh_psnr, unit, (enh_psnr - ori_psnr), unit))
    
    print('SSIM:  ave ori[%d], enh[%d], delta[%d]' % (ori_ssim*10000, enh_ssim*10000, (enh_ssim - ori_ssim)*10000))
    print('> done.')
    print()


if __name__ == '__main__':
    qp_yuv = sorted(glob.glob('./mfqe_datasets/YUV/test_qp_yuv/QP37' + '/*.yuv'))
    raw_yuv = sorted(glob.glob('./mfqe_datasets/YUV/test_raw_yuv' + '/*.yuv'))

    # for test_file_num in range(len(raw_yuv)):
    input_file =  qp_yuv[5]
    label_file = raw_yuv[5]
    print('*' * 70)
    print(input_file)
    print('*' * 70)
    test_one_video(input_file, label_file)
