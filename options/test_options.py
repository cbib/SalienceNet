from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=1000, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        parser.add_argument('--wcrit1', type=float, default=1, help='Weight of criterion 1 in custom loss - MSE')
        parser.add_argument('--wcrit2', type=float, default=0, help='Weight of criterion 2 in custom loss - SSIM')
        parser.add_argument('--wcrit3', type=float, default=0, help='Weight of criterion 3 in custom loss - MGE')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | SSIM | wgangp | LSSSIMGRAD]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')


        self.isTrain = False
        return parser
