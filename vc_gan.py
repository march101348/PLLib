import os
from dotenv import load_dotenv

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

from pllib.sound.data import PairedAudioDataModule
from pllib.image.net.GAN import CycleGANModule
from pllib.sound.utils import SoundSpecTransmitter, SoundIO


def set_encoder_utf8():
    if os.name == 'nt':
        import _locale
        _locale._getdefaultlocale_backup = _locale._getdefaultlocale
        _locale._getdefaultlocale = (lambda *args: (_locale._getdefaultlocale_backup()[0], 'UTF-8'))


def main():
    load_dotenv("vc_gan.env")

    filepath1 = os.getenv('FILEPATH1')
    filepath2 = os.getenv('FILEPATH2')

    g_lr, d_lr = float(os.getenv('G_LR', 1e-4)), float(os.getenv('d_lr', 1e-4))
    batch_size = int(os.getenv('BATCH_SIZE', 8))
    grad_acc = int(os.getenv('GRAD_ACC', 1))
    num_epoch = int(os.getenv('NUM_EPOCH', 30))
    lr = {'G': g_lr, 'D': d_lr}

    sampling_rate = int(os.getenv('SAMPLING_RATE', 16_000))
    span = int(os.getenv('SPAN', 256))
    hop_length = int(os.getenv('HOP_LENGTH', 160))
    n_fft = span * 2 - 2

    sio = SoundIO(sampling_rate=sampling_rate)
    tm = SoundSpecTransmitter(n_fft=n_fft, hop_length=hop_length, to_db=True)
    dm = PairedAudioDataModule(filepath1, filepath2, span=span, batch_size=batch_size, transmitter=tm, transforms=None,
                               sound_io=sio)
    print(tm.mean, tm.std)
    model = CycleGANModule(lr=lr, reconstr_w=5., id_w=1., transmitter=tm, sound_io=sio)
    mlf_logger = MLFlowLogger(
        experiment_name='vc_gan',
        tracking_uri='file:./vc_gan'
    )
    trainer = Trainer(gpus=-1, precision=32, deterministic=True, accumulate_grad_batches=grad_acc,
                      profiler='simple', max_epochs=num_epoch, logger=mlf_logger)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    set_encoder_utf8()
    main()
