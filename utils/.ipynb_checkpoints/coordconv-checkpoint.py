import torch.nn.modules.conv as conv
import torch.nn as nn
import torch

class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, add_coord, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, add_coord, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                            kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out
class AddCoords(nn.Module):
    def __init__(self, rank, add_coord, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda
        self.add_coord = add_coord

    def forward(self, input_tensor):
        if self.add_coord == "-1":
            batch_size_shape, _, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)


            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

            return out
        
        elif self.add_coord == "0":
            # Calculate the center positions
            center_y = 0
            center_x = 0

            # Create y-coordinate map with normalized distances from the center
            y_coord = torch.arange(input_tensor.size(2), dtype=torch.float32)
            y_coord = y_coord / (input_tensor.size(2) - 1)
            y_coord_map = (
                y_coord.view(1, 1, -1, 1).repeat(1, 1, 1, input_tensor.size(3))
            )

            # Create x-coordinate map with normalized distances from the center
            x_coord = torch.arange(input_tensor.size(3), dtype=torch.float32)
            x_coord = x_coord / (input_tensor.size(3) - 1)
            x_coord_map = (
                x_coord.view(1, 1, 1, -1).repeat(1, 1, input_tensor.size(2), 1)
            )
            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                x_coord_map = x_coord_map.cuda()
                y_coord_map = y_coord_map.cuda()
            features_with_pos_enc = torch.cat([input_tensor, x_coord_map, y_coord_map], dim=1)

            return features_with_pos_enc

        