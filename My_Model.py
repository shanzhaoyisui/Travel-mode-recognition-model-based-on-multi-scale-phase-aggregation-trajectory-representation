import torch
import torch.nn as nn
# import os
from ndf import Forest
import copy
import torch.nn.functional as F
from My_Tools import choice_weight
from timm.models.layers import DropPath  # , trunc_normal_
# from timm.models.layers import DropPath


def get_clones(module, num):
    return nn.ModuleList([copy.deepcopy(module) for i in range(num)])


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, (1, 1), (1, 1))
        self.fc2 = nn.Conv2d(hidden_features, out_features, (1, 1), (1, 1))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc'):
        super(PATM, self).__init__()

        self.fc_h = nn.Conv2d(dim, dim, (1, 1), (1, 1), bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, (1, 1), (1, 1), bias=qkv_bias)
        self.fc_c = nn.Conv2d(dim, dim, (1, 1), (1, 1), bias=qkv_bias)

        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=(1, 1), padding=(0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=(1, 1), padding=(7 // 2, 0), groups=dim, bias=False)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, (1, 1), (1, 1), bias=True)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop = nn.Dropout(attn_drop)
        self.mode = mode

        if mode == 'fc':
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, (1, 1), (1, 1), bias=True),
                                              nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, (1, 1), (1, 1), bias=True),
                                              nn.BatchNorm2d(dim), nn.ReLU())
        else:
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, (3, 3), stride=(1, 1),
                                              padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, (3, 3), stride=(1, 1),
                                              padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):

        B, C, H, W = x.shape
        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)

        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)

        #         x_1=self.fc_h(x)
        #         x_2=self.fc_w(x)
        #         x_h=torch.cat([x_1*torch.cos(theta_h),x_2*torch.sin(theta_h)],dim=1)
        #         x_w=torch.cat([x_1*torch.cos(theta_w),x_2*torch.sin(theta_w)],dim=1)

        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attn_Layer(nn.Module):
    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc', num_attn_layer=4):
        super(Attn_Layer, self).__init__()
        # self.norm1 = norm_layer(dim)
        # self.attn = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        self.attn_layer = nn.Sequential(
            norm_layer(dim),
            PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, mode=mode),
            # DropPath(drop_path) if drop_path > 0. else nn.Identity()
        )
        self.num_attn_layer = num_attn_layer
        self.mlp = nn.Sequential(
            norm_layer(dim),
            Mlp(dim, hidden_features=dim * 2),
            # DropPath(drop_path) if drop_path > 0. else nn.Identity()
        )
        # self.heading = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)  # 这里要不要这个MLP？

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))  # torch.Size([64, 4, 16, 60])
        x = x + self.drop_path(self.attn_layer(x))
        x = x + self.drop_path(self.mlp(x))
        # x = x + self.drop_path(self.heading(self.norm2(x)))

        return x


class Local_Convolution(nn.Module):
    def __init__(self, conv_out, fc_out):
        super(Local_Convolution, self).__init__()
        self.Conv_cut_in_8 = get_clones(nn.Sequential(nn.Conv2d(1, conv_out, kernel_size=(2, 9), stride=(1, 1),
                                                                bias=True),
                                                      nn.BatchNorm2d(conv_out), nn.ReLU()), 4)
        self.Conv_cut_in_6 = get_clones(nn.Sequential(nn.Conv2d(1, conv_out, kernel_size=(2, 9), stride=(1, 1),
                                                                bias=True),
                                                      nn.BatchNorm2d(conv_out), nn.ReLU()), 4)
        self.Conv_cut_in_4 = get_clones(nn.Sequential(nn.Conv2d(1, conv_out, kernel_size=(2, 9), stride=(1, 1),
                                                                bias=True),
                                                      nn.BatchNorm2d(conv_out), nn.ReLU()), 4)
        self.full_connection_8 = get_clones(nn.Sequential(
            nn.Linear(conv_out, fc_out, bias=True),
            nn.ReLU()
        ), 4)
        self.full_connection_6 = get_clones(nn.Sequential(
            nn.Linear(conv_out, fc_out, bias=True),
            nn.ReLU()
        ), 4)
        self.full_connection_4 = get_clones(nn.Sequential(
            nn.Linear(conv_out, fc_out, bias=True),
            nn.ReLU()
        ), 4)
        self.conv_out = conv_out
        self.fc_out = fc_out

    def forward(self, seg_cut_in_8, seg_cut_in_6, seg_cut_in_4):
        out_list = []

        for i in range(4):
            fc_in_8 = self.Conv_cut_in_8[i](seg_cut_in_8[:, i, :, :].unsqueeze(1)).squeeze(-1).transpose(1, 2)
            # torch.Size([64, 7, 15])
            fc_in_6 = self.Conv_cut_in_6[i](seg_cut_in_6[:, i, :, :].unsqueeze(1)).squeeze(-1).transpose(1, 2)
            # torch.Size([64, 5, 15])
            fc_in_4 = self.Conv_cut_in_4[i](seg_cut_in_4[:, i, :, :].unsqueeze(1)).squeeze(-1).transpose(1, 2)
            # torch.Size([64, 3, 15])
            out_8 = self.full_connection_8[i](fc_in_8)  # torch.Size([64, 7, 60])
            out_6 = self.full_connection_6[i](fc_in_6)  # torch.Size([64, 5, 60])
            out_4 = self.full_connection_4[i](fc_in_4)  # torch.Size([64, 3, 60])
            out_tmp = torch.cat((out_8, out_6, out_4), dim=1).unsqueeze(1)
            out_list.append(out_tmp)

        out = torch.cat(out_list, dim=1)  # torch.Size([64, 4, 15, 60])

        return out


class Global_Convolution(nn.Module):
    def __init__(self, conv_out, fc_out):
        super(Global_Convolution, self).__init__()
        self.Global_Conv = get_clones(nn.Sequential(nn.Conv2d(1, conv_out, kernel_size=(1, 9), stride=(1, 1)),
                                                    nn.BatchNorm2d(conv_out), nn.ReLU()), 4)
        self.full_connection = get_clones(nn.Sequential(nn.Linear(conv_out, fc_out), nn.ReLU()), 4)

    def forward(self, global_data):
        out_list = []

        for i in range(4):
            fc_in = self.Global_Conv[i](global_data[:, i, :, :].unsqueeze(1)).squeeze(-1).transpose(1, 2)
            # torch.Size([64, 1, 15])
            fc_out = self.full_connection[i](fc_in)  
            out_list.append(fc_out)

        out = torch.cat(out_list, dim=1).unsqueeze(2)  

        return out


class My_Model(nn.Module):
    def __init__(self, conv_out, fc_out, n_tree, tree_depth,
                 n_in_feature, tree_feature_rate, n_class, jointly_training,
                 full_connect_mid_dim_1, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d, mode='fc', feature_num=4, num_attn_layer=4, drop_out=0., mlp_hidden_dim=32):
        super(My_Model, self).__init__()
        self.LC = Local_Convolution(conv_out, fc_out)
        self.GC = Global_Convolution(conv_out, fc_out)
        feature_num_2 = feature_num * 2
        self.Conv_feature = nn.Sequential(
            nn.BatchNorm2d(4),
            nn.Conv2d(4, feature_num, kernel_size=(1, 3), stride=(1, 1), padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(feature_num, feature_num, kernel_size=(1, 3), stride=(1, 1), padding=0, bias=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, fc_out)),
            # nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(feature_num, feature_num_2, kernel_size=(1, 3), stride=(1, 1), padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(feature_num_2, feature_num_2, kernel_size=(1, 3), stride=(1, 1), padding=0, bias=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, fc_out)),
        )
        self.Attn_layer = Attn_Layer(feature_num_2, mlp_ratio,
                                     qkv_bias, qk_scale,
                                     attn_drop, drop_path,
                                     act_layer, norm_layer, mode, num_attn_layer)

        self.Weight_Full_Connect = nn.Sequential(  
            nn.BatchNorm2d(feature_num_2),
            nn.Conv2d(feature_num_2, feature_num, kernel_size=(16, 1), stride=(1, 1), bias=True,
                      dtype=torch.float64),
            nn.ReLU(),
            nn.Conv2d(feature_num, feature_num // 2, kernel_size=(16, 1), stride=(1, 1), bias=True,
                      dtype=torch.float64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, None)),
            nn.Conv2d(feature_num // 2, feature_num // 4, kernel_size=(2, 1), bias=True, stride=(1, 1),
                      dtype=torch.float64),
            nn.ReLU(),
            nn.Conv2d(feature_num // 4, 1, kernel_size=(1, 1), stride=(1, 1), bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, None))
        )

        self.NDF = Forest(n_tree, tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training)


    def forward(self, seg_cut_in_8, seg_cut_in_6, seg_cut_in_4, global_data):
        local_conv_out = self.LC(seg_cut_in_8, seg_cut_in_6, seg_cut_in_4)
        global_conv_out = self.GC(global_data)
        x = torch.cat((local_conv_out, global_conv_out), dim=2)  # x.shape=
        x = self.Conv_feature(x)  # x.shape=
        attn_out = self.Attn_layer(x)  # attn_out.shape=
        wec_out = torch.cat((attn_out, x), dim=2)  # wec_out=
        # wec_out = attn_out + x
        # wec_out = wec_out.view(batch_size, 1, -1)
        # wec_out = wec_out.view(batch_size, 1, channel * H * W)
        wec_out = self.Weight_Full_Connect(wec_out)  # 现在是：wec_out.shape=
        batch_size, channel, H, W = wec_out.shape
        wec_out = wec_out.view(batch_size, channel * H * W)
        out = self.NDF(wec_out)
        # out = self.heading(wec_out)

        return out


def get_model(train_test_opt):
    model = My_Model(train_test_opt['conv_out_dim'], train_test_opt['fc_out_dim'],
                     train_test_opt['n_tree'], train_test_opt['tree_depth'],
                     train_test_opt['n_in_feature'], train_test_opt['tree_feature_rate'],
                     train_test_opt['n_class'], train_test_opt['jointly_training'],
                     train_test_opt['full_connect_mid_dim_1'], train_test_opt['mlp_ratio'],
                     train_test_opt['qkv_bias'], train_test_opt['qk_scale'],
                     train_test_opt['attn_drop'], train_test_opt['drop_path'],
                     train_test_opt['act_layer'], train_test_opt['norm_layer'],
                     train_test_opt['mode'], train_test_opt['feature_num'],
                     train_test_opt['num_attn_layer'], train_test_opt['drop_out'],
                     train_test_opt['mlp_hidden_dim'])
    if train_test_opt['load_weights'] == 1:
        print("loading pretrained weights...")
        load_state_dict = choice_weight(train_test_opt['load_state_dict'])
        model.load_state_dict(torch.load(load_state_dict))
        # model = torch.load(torch.load(load_state_dict))
    elif train_test_opt['load_weights'] == 0:
        for p in model.parameters():
            p = p.to(torch.float64)
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model
