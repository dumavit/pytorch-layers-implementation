from __future__ import print_function
import torch


def diff_mse(x, y):
    x_vec = x.contiguous().view(1, -1).squeeze()
    y_vec = y.contiguous().view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


def conv2d_scalar(x_in, conv_weight, conv_bias, device):
    N, C_in, H, W = x_in.shape
    C_out, _, kernel_size, _ = conv_weight.shape
    image_out_height = H - kernel_size + 1
    image_out_width = W - kernel_size + 1

    # Check convolutional kernel is square
    assert conv_weight.shape[2] == conv_weight.shape[3]

    result = torch.empty(N, C_out, image_out_height, image_out_width).to(device)
    for n in range(N):
        for c_out in range(C_out):
            for m in range(image_out_height):
                for l in range(image_out_width):
                    conv_result = 0
                    for c_in in range(C_in):
                        for i in range(kernel_size):
                            for j in range(kernel_size):
                                conv_result += x_in[n, c_in, m + i, l + j] * conv_weight[c_out, c_in, i, j]

                    result[n, c_out, m, l] = conv_result + conv_bias[c_out]
    return result


def conv2d_vector(x_in, conv_weight, conv_bias, device):
    N, C_in, H, W = x_in.shape
    C_out, _, kernel_size, _ = conv_weight.shape
    image_out_height = H - kernel_size + 1
    image_out_width = W - kernel_size + 1

    x_col = im2col(x_in, kernel_size, device)
    conv_weight_rows = conv_weight2rows(conv_weight)

    # Perform w*x + b
    result = conv_weight_rows.mm(x_col).add(conv_bias.view((C_out, 1)))
    # Calculate sum over channels_in dimension
    result = result.view((C_out, N, C_in, image_out_height, image_out_width)).sum(dim=2).transpose(0, 1)
    return result


def im2col(img, kernel_size, device, stride=1):
    N, C, H, W = img.shape

    out_height = (H - kernel_size) // stride + 1
    out_width = (W - kernel_size) // stride + 1
    col = torch.zeros((kernel_size, kernel_size, N, C, out_height, out_width)).to(device)

    for y in range(kernel_size):
        y_max = y + stride * out_height
        for x in range(kernel_size):
            x_max = x + stride * out_width
            col[y, x, :, :, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    return col.view((kernel_size ** 2, -1))


def conv_weight2rows(conv_weight):
    c_out, c_in = conv_weight.shape[0:2]
    return conv_weight.clone().view((c_out * c_in, -1))


def pool2d_scalar(a, device):
    N, C, H, W = a.shape

    pooling_size = 2
    image_out_heigth = H // pooling_size
    image_out_width = W // pooling_size
    result = torch.empty(N, C, image_out_heigth, image_out_width).to(device)

    for n in range(N):
        for c in range(C):
            for i in range(image_out_heigth):
                for j in range(image_out_width):
                    result[n, c, i, j] = max(a[n, c, 2 * i, 2 * j],
                                             a[n, c, 2 * i, 2 * j + 1],
                                             a[n, c, 2 * i + 1, 2 * j],
                                             a[n, c, 2 * i + 1, 2 * j + 1])

    return result


def pool2d_vector(a, device):
    N, C, H, W = a.shape
    pooling_size = 2

    return im2col(a, pooling_size, device, stride=2) \
        .max(dim=0)[0] \
        .view(N, C, H // pooling_size, W // pooling_size)


def relu_scalar(a, device):
    N, size = a.shape
    result = torch.empty((N, size)).to(device)

    for n in range(N):
        for j in range(size):
            result[n, j] = max(a[n, j], 0)
    return result


def relu_vector(a, device):
    result = a.clone()
    result[a < 0] = 0
    return result


def reshape_vector(a, device):
    return a.clone().view((a.shape[0], -1))


def reshape_scalar(a, device):
    N, C_in, M, L = a.shape
    result = torch.empty((N, C_in * M * L)).to(device)

    for n in range(N):
        for c in range(C_in):
            for m in range(M):
                for l in range(L):
                    j_index = c * M * M + m * M + l
                    result[n, j_index] = a[n, c, m, l]
    return result


def fc_layer_scalar(a, weight, bias, device):
    N = a.shape[0]
    output_size, input_size = weight.shape
    z = torch.empty(N, output_size).to(device)

    for n in range(N):
        for j in range(output_size):
            z[n, j] = bias[j]
            for i in range(input_size):
                z[n, j] += weight[j, i] * a[n, i]
    return z


def fc_layer_vector(a, weight, bias, device):
    return a.mm(weight.t()).add(bias)
