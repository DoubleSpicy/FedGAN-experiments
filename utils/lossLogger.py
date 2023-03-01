import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pytorch_fid.fid_score as fid

class logger():
    def __init__(self, dir: str, id: int, columns: list, x_label: str, y_label: str) -> None:
        assert len(columns) > 0
        self.data = pd.DataFrame(columns=columns)
        self.columns = columns
        self.dir = Path(dir)
        self.csv_path = Path(dir).joinpath()
        self.id = id
        self.x_label = x_label
        self.y_label = y_label

    def _write(self):
        # write the df
        self.data.to_csv(self._get_path(self.y_label + str(self.id) + ".csv"))
        self.draw()

    def _get_path(self, fileName):
        return self.dir.joinpath(fileName)

    def concat(self, values: list):
        # concat one row to df
        self.data.loc[len(self.data.index)] = values
        self._write()

    def draw(self):
        # df = pd.read_csv('/mnt/e/ml/gan/runs/_WGAN-GP_0.4_True_CelebA_AvgMod_1_delay_/lossLog0.csv')
        points = [self.data[col] for col in self.columns]
        x = [i for i in range(1, len(self.data[self.columns[0]])+1)]
        for col in self.columns:
            plt.plot(x, self.data[col], label=col)
        plt.title(f'{self.y_label} of client ID {self.id}')
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend(loc='best')
        plt.savefig(self._get_path(f'{self.y_label}{self.id}.png'))
        plt.clf()

class lossLogger(logger):
    def __init__(self, dir: str, id: int, columns: list, x_label: str, y_label: str) -> None:
        super().__init__(dir, id, columns, x_label, y_label)

    def _copy_to_cpu(self, x):
        return x.detach().clone().cpu()

    def concat(self, values: list):
        # concat one row to df
        self.data.loc[len(self.data.index)] = [self._copy_to_cpu(val).item() for val in values]
        self._write()

class FIDLogger(logger):
    def __init__(self, dir: str, id: int, x_label: str, y_label: str, columns: list = ['fid']) -> None:
        super().__init__(dir, id, columns, x_label, y_label)

    def cal_FID_score(self, path1):
        self.concat(fid.main([path1, self.npz]))
        self.draw()

if __name__ == '__main__':
    log = FIDLogger(dir='./', x_label='testX', y_label='testY', columns=['loss1','loss2','loss3'], id=0)
    for i in range(5):
        log.concat([i+j for j in range(3)])
    