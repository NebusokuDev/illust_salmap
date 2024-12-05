import unittest
import torch


class MyTestCase(unittest.TestCase):
    def test_encoder_resolution_reduction(self):
        input_shape = (4, 3, 64, 64)
        batch, in_ch, width, height = input_shape
        out_ch = 64

        input_noise = torch.randn(input_shape)
        output_shape = (batch, out_ch, width // 2, height // 2)

        encoder = Encoder()
        self.assertEqual(encoder(input_noise).shape, output_shape)

    def test_decoder_resolution_upsample(self):
        input_shape = (4, 3, 64, 64)
        batch, in_ch, width, height = input_shape
        out_ch = 32
        input_noise = torch.randn(input_shape)
        output_shape = (batch, out_ch, width * 2, height * 2)

        encoder = Encoder()
        self.assertEqual(encoder(input_noise).shape, output_shape)

    def test_unet_input_output_size(self):
        input_shape = (4, 3, 64, 64)
        output_shape = (4, 1, 64, 64)
        model = UNet(out_ch=1)

        output = model(torch.randn(*input_shape))

        self.assertEqual(output.shape, output_shape)


if __name__ == '__main__':
    unittest.main()
