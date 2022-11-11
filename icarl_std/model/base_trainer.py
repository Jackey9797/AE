from collections.abc import Callable, Sequence
from model.ae import AutoEncoder
from torch import nn, Tensor
from typing import TypeVar
from util.normalization import Normalizer

import sys
import torch


class BaseTrainer():

    dataset_name = 'cert4.2'
    a = 0.2
    a_threhold = 0.1

    model: AutoEncoder
    normalizer: Normalizer
    device: str
    weekly_target: float
    init_epoch: int
    fine_tuning_epoch: int
    init_lr: float
    tuning_lr: float

    # def gen_label_dist_png():
    #     arr = torch.zeros(size=(1000, len(all_data)), dtype=torch.int)
    #     for week_id in range(len(all_data)):
    #         user, x, y = all_data[week_id]
    #         for user_id in range(1000):
    #             arr[user_id, week_id] = torch.sum(y[user == user_id])
    #     from PIL import Image
    #     arr = torch.stack([l for l in arr if l.any()])
    #     arr = arr[arr.sum(1).sort()[1]]
    #     arr = arr[torch.stack([l.nonzero()[0] for l in arr]).squeeze().sort()[1]]
    #     arrr = (arr/7.*255).cpu().numpy().astype('uint8')
    #     im = Image.fromarray(arrr)
    #     im.save('t.png')
    # gen_label_dist_png()

    Tensors = TypeVar('Tensors', bound=Sequence[Tensor])

    @staticmethod
    def cat_by_range(all_data: dict[int, Tensors], week_range: Sequence[int]) -> Tensors:
        result = [torch.cat([all_data[week][i] for week in week_range]) for i in range(len(all_data[week_range[0]]))]
        return result

    def tensor_getter(self, mode: str):
        from feature import CertFeature
        from util.config import default_config
        from util.spark import F
        config = default_config.dataset_config
        cert = CertFeature(input_path=config.input_path,
                           answers_path=config.answers_path,
                           output_path=config.output_path,
                           mode=mode,
                           version=config.version)
        df_cert = cert.get().filter(F.col('week_id') != 0).filter(F.col('week_id') != 72).toPandas()
        removed_cols = ['user', 'day', 'week_id', 'starttime', 'endtime', 'sessionid', 'insider']
        x_cols = [i for i in df_cert.columns if i not in removed_cols]
        week_end: int = max(df_cert['week_id'])-1
        result: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
        for i in range(week_end):
            df_week = df_cert[df_cert['week_id'] == i+1]
            user = torch.tensor(df_week['user'].values, dtype=torch.float32, device=self.device)
            x = torch.tensor(df_week[x_cols].values, dtype=torch.float32, device=self.device)
            y = torch.tensor(df_week['insider'].values.astype('bool'), dtype=torch.bool, device=self.device)
            result |= {i: (user, x, y)}
        return result

    def printe(self, msg: str):
        print(f'{msg}, x: {(self.weekly_target, self.init_epoch, self.fine_tuning_epoch, self.init_lr, self.tuning_lr)}', file=sys.stderr, flush=True)

    @staticmethod
    def weakly_loss(re_error: Tensor, label: Tensor, a: float):
        assert re_error.shape == label.shape
        return torch.mean((~label)*re_error+label*nn.functional.relu(a-re_error))

    @staticmethod
    def train_step(loss_func: Callable[[], Tensor], optimizer: torch.optim.Optimizer):
        loss = loss_func()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.unsqueeze(-1)

    @torch.no_grad()
    def test(self, x: Tensor, y: Tensor):
        self.model.eval()
        x_test = self.normalizer.normalize(x)

        x_test_pred: Tensor = self.model(x_test)
        re_error: Tensor = (x_test - x_test_pred).pow(2).mean(dim=1)
        sort_indices: Tensor
        _, sort_indices = torch.sort(re_error, descending=True)
        # get best threshold
        P = len(y.nonzero())
        N = len(y)-P
        # assert P != 0
        sorted_y = y[sort_indices]
        y_array = sorted_y.repeat(len(y)+1, 1)
        possible_tp = y_array.tril(-1).sum(dim=-1)
        possible_fp = torch.arange(len(y)+1, device=self.device)-possible_tp
        possible_tpr = possible_tp/P
        possible_fpr = possible_fp/N
        best_i = torch.argmax(possible_tpr-possible_fpr)  # 0 <= best_i <= len(y)
        abnormal_indices = sort_indices[:int(best_i)]
        t = torch.zeros_like(y)
        t[abnormal_indices] = True
        TP = torch.sum(t & y)
        FP = torch.sum(t & ~y)
        FN = torch.sum(~t & y)
        top_a = sort_indices[:int(len(sort_indices)*self.a)]
        return FP, TP, FN, top_a
