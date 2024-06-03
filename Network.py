import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.dcn.deform_conv import ModulatedDeformConv

# ==========
# Mutile-level resiudual deformable fusion module
# ==========
class Res_Block(nn.Module):
    def __init__(self, in_nc, bks=3):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=bks, stride=1, padding=bks // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=bks, stride=1, padding=bks // 2)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return self.relu(x + res)
        

class Down_layer(nn.Module):
    def __init__(self, in_nc, bks=3):
        super(Down_layer, self).__init__()
        self.d_conv = nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=bks, stride=2, padding=bks // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=bks, stride=1, padding=bks // 2)

    def forward(self, x):
        x = self.d_conv(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        return x


class Up_layer(nn.Module):
    def __init__(self, in_nc, bks=3):
        super(Up_layer, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels=in_nc, out_channels=in_nc, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=bks, stride=1, padding=bks // 2)

    def forward(self, x):
        x = self.up_conv(x)
        x = self.relu(x)
        x = self.conv1(x)
        return x


class MLRD(nn.Module):  
    def __init__(self, in_nc, m_nc, out_nc, t=7, bks=3, dks=3):
        super(MLRD, self).__init__()
        self.in_nc = in_nc
        self.d_size = dks * dks
        self.conv_first = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=bks, stride=1, padding=bks // 2), 
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=bks, stride=1, padding=bks // 2), 
            nn.ReLU(inplace=True)
        )
        self.conv_cat1 = nn.Conv2d(in_channels=32 * t, out_channels=m_nc, kernel_size=1, stride=1, padding=0)


        self.down_conv1 = Down_layer(in_nc=m_nc, bks=bks)
        self.down_conv2 = Down_layer(in_nc=m_nc, bks=bks)
        self.down_conv3 = Down_layer(in_nc=m_nc, bks=bks)
        self.rse1 = Res_Block(in_nc=m_nc, bks=3)
        self.rse2_1 = Res_Block(in_nc=m_nc, bks=3)
        self.rse2_2 = Res_Block(in_nc=m_nc, bks=3)
        self.rse3_1 = Res_Block(in_nc=m_nc, bks=3)
        self.rse3_2 = Res_Block(in_nc=m_nc, bks=3)
        self.trans_conv3 = nn.ConvTranspose2d(in_channels=m_nc, out_channels=m_nc, kernel_size=4, stride=2, padding=1)
        self.trans_conv2 = nn.ConvTranspose2d(in_channels=m_nc, out_channels=m_nc, kernel_size=4, stride=2, padding=1)
        self.up_conv1 = Up_layer(in_nc=m_nc, bks=3)
        self.up_conv2 = Up_layer(in_nc=m_nc, bks=3)
        self.up_conv3 = Up_layer(in_nc=m_nc, bks=3)
        self.conv_add = nn.Conv2d(in_channels=m_nc, out_channels=m_nc, kernel_size=bks, stride=1, padding=bks // 2)

        self.offset_mask = nn.Conv2d(m_nc, in_nc * 3 * self.d_size, bks, padding=bks // 2)
        self.deform_conv = ModulatedDeformConv(in_nc, out_nc, dks, padding=dks // 2, deformable_groups=in_nc)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        b, c, t, h, w = x.size()  # b,1, t, h, w
        x1 = self.conv_first(x)

        x1 = x1.permute(0, 2, 1, 3, 4).contiguous().view(b, -1, h, w)
        x1 = self.conv_cat1(x1)
        x1 = self.relu(x1)

        x3 = self.down_conv2(x1)
        x3 = self.down_conv3(x3)
        x3 = self.rse3_1(x3)
        
        x2 = self.down_conv1(x1)
        x2 = x2 + self.trans_conv3(x3)
        x2 = self.rse2_1(x2)

        x1 = x1 + self.trans_conv2(x2)
        x1 = self.rse1(x1)

        x2 = self.rse2_2(x2)
        x2 = self.up_conv1(x2)

        x3 = self.rse3_2(x3)
        x3 = self.up_conv2(x3)
        x3 = self.relu(x3)
        x3 = self.up_conv3(x3)

        out = x1 + x2 + x3 
        out = self.conv_add(out)
        out = self.relu(out)

        off_mask = self.offset_mask(out)
        off = off_mask[:, :self.in_nc * 2 * self.d_size, ...]  
        mask = torch.sigmoid(off_mask[:, self.in_nc * 2 * self.d_size:, ...]) 
        # perform deformable convolutional fusion
        x = x.squeeze(1)
        fused_feat = F.relu(self.deform_conv(x, off, mask), inplace=True)
        return fused_feat


class DS_net(nn.Module):
    def __init__(self, in_channel):
        super(DS_net, self).__init__()
        self.conv3_1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)
        self.conv7_1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)

        self.conv_c = nn.Conv2d(in_channels=in_channel*4, out_channels=in_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = F.relu(x)
        x1 = self.conv3_1(x)
        x1 = self.relu(x1)
        x2 = self.conv5_1(x1)
        x2 = self.relu(x2)
        x3 = self.conv7_1(x2)
        x3 = self.relu(x3)

        x_cat = torch.cat((x, x1, x2, x3), 1)
        x_cat = self.conv_c(x_cat)
        return x_cat


class QE_net(nn.Module):   
    def __init__(self, in_channel, m_channel, out_channel, bks=3):
        super(QE_net, self).__init__()
        self.conv_first = nn.Conv2d(in_channels=in_channel, out_channels=m_channel, kernel_size=bks, stride=1, padding=bks // 2)

        self.DS_net1 = DS_net(in_channel=m_channel)
        self.DS_net2 = DS_net(in_channel=m_channel)
        self.DS_net3 = DS_net(in_channel=m_channel)

        self.conv_c = nn.Conv2d(in_channels=m_channel * 4, out_channels=m_channel, kernel_size=bks, stride=1, padding=bks // 2)
        self.conv = nn.Conv2d(in_channels=m_channel, out_channels=m_channel, kernel_size=bks, stride=1, padding=bks // 2)
        self.conv_last = nn.Conv2d(in_channels=m_channel, out_channels=out_channel, kernel_size=bks, stride=1, padding=bks // 2) 
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv_first(x)

        ds1 = self.DS_net1(x)
        ds2 = x - ds1
        ds2 = self.DS_net2(ds2)
        ds2 = x + ds2
        ds3 = self.DS_net3(ds2)

        ds_c = torch.cat((x, ds1, ds2, ds3), 1)
        ds_c = self.conv_c(ds_c)
        out = ds_c + x
        out = self.conv(out)
        out = self.relu(out)
        out = self.conv_last(out)
        return out


# ==========
# Network
# ==========
class Net(nn.Module):
    """
    in: (B T*C H W)
    out: (B C H W)
    """
    def __init__(self, opts_dict):
        super(Net, self).__init__()
        self.radius = opts_dict['radius']
        self.t = 2 * self.radius + 1
        self.in_nc = opts_dict['mlrd']['in_nc']
        
        self.mlrd = MLRD(in_nc=self.in_nc * self.t, m_nc=opts_dict['mlrd']['m_nc'], out_nc=opts_dict['mlrd']['out_nc'], t=self.t, 
                    bks=opts_dict['mlrd']['bks'], dks=opts_dict['mlrd']['dks'])
        self.qe = QE_net(in_channel=opts_dict['qe']['in_nc'], m_channel=opts_dict['qe']['m_nc'], out_channel=opts_dict['qe']['out_nc'], 
                            bks=opts_dict['qe']['bks'])

    def forward(self, x):
        x = x.unsqueeze(1)  # b, c, t, h, w

        out = self.mlrd(x)
        out = self.qe(out)

        x = x.squeeze(1)
        
        frm_lst = [self.radius + idx_c * self.t for idx_c in range(self.in_nc)]
        out += x[:, frm_lst, ...]  # res: add middle frame
        return out




