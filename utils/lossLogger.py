import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

class lossLogger():
    def __init__(self, dir: str, id: int) -> None:
        self.data = pd.DataFrame(columns=['d_loss_real', 'd_loss_fake', 'g_loss'])
        self.dir = Path(dir)
        self.csv_path = Path(dir).joinpath()
        self.id = id
        # self._write()

    def _write(self):
        # write the df
        self.data.to_csv(self._get_path('lossLog' + str(self.id) + ".csv"))
        self.draw_loss_plot()

    def _copy_to_cpu(self, x):
        return x.detach().clone().cpu()

    def _get_path(self, fileName):
        return self.dir.joinpath(fileName)

    def concat(self, loss_real, loss_fake, g_loss):
        loss_real, loss_fake, g_loss = self._copy_to_cpu(loss_real), self._copy_to_cpu(loss_fake), self._copy_to_cpu(g_loss)
        # concat one row to df
        self.data.loc[len(self.data.index)] = [loss_real.item(), loss_fake.item(), g_loss.item()]
        self._write()

    def draw_loss_plot(self):
        # df = pd.read_csv('/mnt/e/ml/gan/runs/_WGAN-GP_0.4_True_CelebA_AvgMod_1_delay_/lossLog0.csv')
        d_loss_fake = self.data['d_loss_fake']
        d_loss_real = self.data['d_loss_real']
        g_loss = self.data['g_loss']
        x = [i for i in range(1, len(self.data['d_loss_fake'])+1)]
        plt.plot(x, d_loss_fake, label='d_loss_fake')
        plt.plot(x, d_loss_real, label='d_loss_real')
        plt.plot(x, g_loss, label='g_loss')
        plt.title(f'loss of client ID {self.id}')
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.savefig(self._get_path(f'loss{self.id}.png'))
        plt.clf()


if __name__ == '__main__':
    log = lossLogger('/mnt/e/ml/gan/', '0')
    log.draw_loss_plot()