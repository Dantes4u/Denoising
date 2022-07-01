import torch
import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channe;s
        :param kernel_size: kernel size
        :param stride: stride
        :param batch_norm: include a BN layer?
        :param activation: Type of activation; None if none
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # A container that will hold the layers in this convolutional block
        layers = list()

        # A convolutional layer
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # A batch normalization (BN) layer, if wanted
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # An activation layer, if wanted
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # Put together the convolutional block as a sequence of the layers in this container
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, in_channels, w, h)
        :return: output images, a tensor of size (N, out_channels, w, h)
        """
        output = self.conv_block(input)  # (N, out_channels, w, h)

        return output


class SubPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """

    def __init__(self, kernel_size=3, in_channels=64, out_channels=64, scaling_factor=2, stride=(1, 2)):
        """
        :param kernel_size: kernel size of the convolution
        :param n_channels: number of input and output channels
        :param scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # A container that will hold the layers in this convolutional block
        layers = list()

        # A convolutional layer that increases the number of out channels by scaling factor^2, followed by pixel shuffle, batch normalization and PReLU
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2, stride=stride)
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)
        output = self.pixel_shuffle(output)
        output = self.batch_norm(output)
        output = self.prelu(output)

        return output


class DenseBlock(nn.Module):
    """
    A dense block consists of five convolutional layers in which the input to a given layer is
    the concatenation of the outputs from all the previous layers in the block
    """

    def __init__(self, kernel_size=3, n_channels=64):
        """
        :param kernel_size: kernel size
        :param n_channels: number of input and output channels (same because the input must be added to the output)
        """
        super(DenseBlock, self).__init__()

        self.conv_1 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv2d(2 * n_channels, n_channels, kernel_size, padding=kernel_size // 2)
        self.conv_3 = nn.Conv2d(3 * n_channels, n_channels, kernel_size, padding=kernel_size // 2)
        self.conv_4 = nn.Conv2d(4 * n_channels, n_channels, kernel_size, padding=kernel_size // 2)
        self.conv_5 = nn.Conv2d(5 * n_channels, n_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: output images, a tensor of size (N, n_channels, w, h)
        """

        conv_1_out = self.conv_1(input)
        conv_2_in = torch.cat([input, conv_1_out], dim=1)

        conv_2_out = self.conv_2(conv_2_in)
        conv_3_in = torch.cat([conv_2_in, conv_2_out], dim=1)

        conv_3_out = self.conv_3(conv_3_in)
        conv_4_in = torch.cat([conv_3_in, conv_3_out], dim=1)

        conv_4_out = self.conv_4(conv_4_in)
        conv_5_in = torch.cat([conv_4_in, conv_4_out], dim=1)

        output = self.conv_5(conv_5_in)

        return output


class DCRN_net(nn.Module):
    def __init__(self):
        super(DCRN_net, self).__init__()

        # Encoder
        self.conv_1 = ConvolutionalBlock(in_channels=1, out_channels=32, kernel_size=5, stride=(2, 1), batch_norm=True,
                                         activation='prelu')
        self.dense_1 = DenseBlock(kernel_size=3, n_channels=32)

        self.conv_2 = ConvolutionalBlock(in_channels=32, out_channels=32, kernel_size=3, stride=(2, 1), batch_norm=True,
                                         activation='prelu')
        self.dense_2 = DenseBlock(kernel_size=3, n_channels=32)

        self.conv_3 = ConvolutionalBlock(in_channels=32, out_channels=32, kernel_size=3, stride=(2, 1), batch_norm=True,
                                         activation='prelu')
        self.dense_3 = DenseBlock(kernel_size=3, n_channels=32)

        self.conv_4 = ConvolutionalBlock(in_channels=32, out_channels=32, kernel_size=3, stride=(2, 1), batch_norm=True,
                                         activation='prelu')
        self.dense_4 = DenseBlock(kernel_size=3, n_channels=32)

        self.conv_5 = ConvolutionalBlock(in_channels=32, out_channels=64, kernel_size=3, stride=(2, 1), batch_norm=True,
                                         activation='prelu')
        self.dense_5 = DenseBlock(kernel_size=3, n_channels=64)

        self.conv_6 = ConvolutionalBlock(in_channels=64, out_channels=128, kernel_size=3, stride=(2, 1),
                                         batch_norm=True, activation='prelu')
        self.conv_7 = ConvolutionalBlock(in_channels=128, out_channels=256, kernel_size=3, stride=(2, 1),
                                         batch_norm=True, activation='prelu')
        self.conv_8 = ConvolutionalBlock(in_channels=256, out_channels=512, kernel_size=3, stride=(2, 1),
                                         batch_norm=True, activation='prelu')

        # LSTM

        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True)

        # Decoder
        self.sbconv_1 = SubPixelConvolutionalBlock(kernel_size=3, in_channels=1024, out_channels=256, stride=(1, 2))
        self.sbconv_2 = SubPixelConvolutionalBlock(kernel_size=3, in_channels=512, out_channels=128, stride=(1, 2))

        self.sbconv_3 = SubPixelConvolutionalBlock(kernel_size=3, in_channels=256, out_channels=64, stride=(1, 2))
        self.dense_6 = DenseBlock(kernel_size=3, n_channels=128)

        self.sbconv_4 = SubPixelConvolutionalBlock(kernel_size=3, in_channels=128, out_channels=32, stride=(1, 2))
        self.dense_7 = DenseBlock(kernel_size=3, n_channels=64)

        self.sbconv_5 = SubPixelConvolutionalBlock(kernel_size=3, in_channels=64, out_channels=32, stride=(1, 2))
        self.dense_8 = DenseBlock(kernel_size=3, n_channels=64)

        self.sbconv_6 = SubPixelConvolutionalBlock(kernel_size=3, in_channels=64, out_channels=32, stride=(1, 2))
        self.dense_9 = DenseBlock(kernel_size=3, n_channels=64)

        self.sbconv_7 = SubPixelConvolutionalBlock(kernel_size=3, in_channels=64, out_channels=32, stride=(1, 2))
        self.dense_10 = DenseBlock(kernel_size=3, n_channels=64)

        self.sbconv_8 = SubPixelConvolutionalBlock(kernel_size=5, in_channels=64, out_channels=1, stride=(1, 2))

    def forward(self, input):
        # Encoder
        encoder_out_1 = self.conv_1(input)
        encoder_out_1 = self.dense_1(encoder_out_1)

        encoder_out_2 = self.conv_2(encoder_out_1)
        encoder_out_2 = self.dense_2(encoder_out_2)

        encoder_out_3 = self.conv_3(encoder_out_2)
        encoder_out_3 = self.dense_3(encoder_out_3)

        encoder_out_4 = self.conv_4(encoder_out_3)
        encoder_out_4 = self.dense_4(encoder_out_4)

        encoder_out_5 = self.conv_5(encoder_out_4)
        encoder_out_5 = self.dense_5(encoder_out_5)

        encoder_out_6 = self.conv_6(encoder_out_5)
        encoder_out_7 = self.conv_7(encoder_out_6)
        encoder_out_8 = self.conv_8(encoder_out_7)

        # LSTM
        encoder_out_8 = encoder_out_8.view(encoder_out_8.size(0), -1, encoder_out_8.size(1))
        encoder_out_8 = encoder_out_8.permute((1, 0, 2))

        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(encoder_out_8)

        lstm_out = lstm_out.view(lstm_out.shape + (1,))
        lstm_out = lstm_out.permute((1, 2, 3, 0))

        encoder_out_8 = encoder_out_8.view(encoder_out_8.shape + (1,))
        encoder_out_8 = encoder_out_8.permute((1, 2, 3, 0))

        decoder_input = torch.cat([encoder_out_8, lstm_out], dim=1)

        # Decoder
        decoder_out_1 = self.sbconv_1(decoder_input)
        decoder_out_1 = torch.cat([encoder_out_7, decoder_out_1], dim=1)

        decoder_out_2 = self.sbconv_2(decoder_out_1)
        decoder_out_2 = torch.cat([encoder_out_6, decoder_out_2], dim=1)

        decoder_out_3 = self.sbconv_3(decoder_out_2)
        decoder_out_3 = torch.cat([encoder_out_5, decoder_out_3], dim=1)
        decoder_out_3 = self.dense_6(decoder_out_3)

        decoder_out_4 = self.sbconv_4(decoder_out_3)
        decoder_out_4 = torch.cat([encoder_out_4, decoder_out_4], dim=1)
        decoder_out_4 = self.dense_7(decoder_out_4)

        decoder_out_5 = self.sbconv_5(decoder_out_4)
        decoder_out_5 = torch.cat([encoder_out_3, decoder_out_5], dim=1)
        decoder_out_5 = self.dense_8(decoder_out_5)

        decoder_out_6 = self.sbconv_6(decoder_out_5)
        decoder_out_6 = torch.cat([encoder_out_2, decoder_out_6], dim=1)
        decoder_out_6 = self.dense_9(decoder_out_6)

        decoder_out_7 = self.sbconv_7(decoder_out_6)
        decoder_out_7 = torch.cat([encoder_out_1, decoder_out_7], dim=1)
        decoder_out_7 = self.dense_10(decoder_out_7)

        output = self.sbconv_8(decoder_out_7)

        return output