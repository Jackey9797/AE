from __future__ import annotations
from collections.abc import Iterable
from model.ae import AutoEncoder
from model.base_trainer import BaseTrainer
from torch import nn, Tensor
from typing import Optional
from util.normalization import Normalizer
from util.tensor import TensorStorage

import numpy as np
import torch

class ExemplarSet():
    def __init__(self, device: str) -> None:
        self.device = device
        self._normal_set: dict[int, Tensor] = {}
        self._abnormal_set: Tensor = torch.empty(size=(0,), device=device)

    def get_normal(self, key: int):
        return self._normal_set.__getitem__(key)

    def add_normal(self, key: int, value: Tensor):
        if self._normal_set.get(key) is not None:
            raise ValueError('key already exist')
        self._normal_set.__setitem__(key, value)

    def add_abnormal(self, value: Tensor):
        self._abnormal_set = torch.cat([self._abnormal_set, value])

    def reduce_size(self, size: int):
        new_range = torch.arange(size)  # release mem insted of using slice view
        for k in self._normal_set:
            old = self._normal_set[k]
            self._normal_set[k] = old[new_range]
        if len(self._normal_set)*size < len(self._abnormal_set):
            new_abnormal_range = torch.arange(len(self._normal_set)*size)
            self._abnormal_set = self._abnormal_set[new_abnormal_range]

    def normal(self):
        yield from self._normal_set.items()

    def abnormal(self):
        return self._abnormal_set


class IcarlTrainer(BaseTrainer):

    def __init__(self,
                 mode: str,
                 init_week_range: Iterable[int],
                 turning_week_range: Iterable[int],
                 K: int,
                 norm_weight: float,
                 init_lr: float = 0.01,
                 tuning_lr: float = 0.01,
                 weekly_target: float = 5.,
                 init_epoch: int = 25,
                 fine_tuning_epoch: int = 15,
                 seed: Optional[int] = None,
                 device: str = 'cuda'
                 ) -> None:
        super().__init__()
        self.all_data = TensorStorage(f'feature-{self.dataset_name}-{mode}.dat', lambda: self.tensor_getter(mode)).get(device)
        self.n_per_week = next(len(x) for _, x, _ in self.all_data.values())
        self.init_week_range = list(init_week_range)
        self.turning_week_range = list(turning_week_range)
        self.K = K
        self.norm_weight = norm_weight
        self.init_epoch = init_epoch
        self.fine_tuning_epoch = fine_tuning_epoch
        self.init_lr = init_lr
        self.tuning_lr = tuning_lr
        self.weekly_target = weekly_target
        self.device = device
        self.normalizer: Normalizer
        self.model: AutoEncoder
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.n_classes = 1
        self.exemplar_sets: ExemplarSet = ExemplarSet(device)  # list of indices

    def init_train(self, x: Tensor, epoch: int):
        x_train = self.normalizer.normalize(x)

        self.model = AutoEncoder(in_features=x_train.shape[1], device=self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr)

        self.model.train()

        def loss_func(): return nn.functional.mse_loss(x_train, self.model(x_train))
        train_losses = [self.train_step(loss_func, optimizer) for _ in range(epoch)]
        avg_loss = torch.cat(train_losses).float().mean()
        self.printe(f'Total loss: {avg_loss:.3f}')

    @staticmethod
    def ceil(a: int, b: int):
        '''
        math.ceil(a/b) without overflow
        '''
        return (a-1)//b+1

    @staticmethod
    @torch.no_grad()
    def balance(x: Tensor, y: Tensor):
        abnormal_indices = y.nonzero().reshape(-1)
        n_abnormal = len(abnormal_indices)
        n_normal = len(x)-n_abnormal
        if n_abnormal == 0 or n_normal == 0 or n_abnormal >= n_normal:
            return x, y
        normal_indices = (y == 0).nonzero().reshape(-1)
        n_repeat = IcarlTrainer.ceil(n_normal, n_abnormal)
        abnormal_indices = abnormal_indices.repeat(n_repeat)[:n_normal]
        indices = torch.cat((normal_indices, abnormal_indices))
        indices = indices[torch.randperm(len(indices))]
        return x[indices], y[indices]

    def incremental_train(self,
                          new_x: Tensor,
                          new_y: Tensor,
                          epoch: int,
                          weakly_target: float):
        self.n_classes += 1
        new_x = self.normalizer.normalize(new_x)
        new_x, new_y = self.balance(new_x, new_y)
        # get all old data [1d]
        old_x = torch.cat(
            *(sample for _, sample in self.exemplar_sets.normal()),
            self.exemplar_sets.abnormal()
        ])
        old_y = torch.cat([
            torch.zeros(size=(len(old_x)-len(self.exemplar_sets.abnormal()),)),
            torch.ones(size=(len(self.exemplar_sets.abnormal()),)),
        ])
        # balance old normal data and abnormal data
        old_x, old_y = self.balance(old_x, old_y)

        self.model.eval()
        with torch.no_grad():
            old_score = self.model.get_score(old_x)
            score_normalizer = Normalizer(old_score)
            old_score = score_normalizer.normalize_nonzero(old_score)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.tuning_lr)
        self.model.train()
        for _ in range(epoch):
            def loss_func():
                re_error: Tensor = self.model.get_score(new_x)
                loss1 = self.weakly_loss(re_error, new_y, a=weakly_target)
                new_score = self.model.get_score(old_x)
                new_score = score_normalizer.normalize_nonzero(new_score)
                loss2 = torch.nn.functional.mse_loss(new_score, old_score)
                return loss1+self.norm_weight*loss2
            self.train_step(loss_func, optimizer)

    @torch.no_grad()
    def create_new_class_set(self, x: Tensor, y: Tensor, n_per_class: int, _class: int):
        new_normal = x[(~y).nonzero().reshape(-1)]
        new_abnormal = x[y.nonzero().reshape(-1)]
        self.exemplar_sets.add_abnormal(new_abnormal)
        if n_per_class > len(new_normal):
            self.exemplar_sets.add_normal(_class, new_normal)
            return
        self.model.eval()
        normal_score = self.model.get_score(new_normal)
        normal_score = Normalizer.normalize_nonzero_(normal_score)
        normal_score_filtered = normal_score.clone()
        selected_normal_indices = torch.empty(size=(n_per_class,), dtype=torch.long, device=self.device)
        for k in range(n_per_class):
            current_selected_sum = normal_score[selected_normal_indices[:k]].sum()
            target = torch.argmin(torch.abs_((normal_score_filtered+current_selected_sum)/(k+1)))
            selected_normal_indices[k] = target
            normal_score_filtered[target] /= 0
        self.exemplar_sets.add_normal(_class, new_normal[selected_normal_indices])

    def run(self):
        _, self.init_x, _ = self.cat_by_range(self.all_data, self.init_week_range)
        self.normalizer = Normalizer(self.init_x)
        self.init_train(self.init_x, epoch=self.init_epoch)

        init_x = self.init_x[torch.randperm(len(self.init_x))[:self.K]]
        self.exemplar_sets.add_normal(-1, init_x)

        def test_and_tuing(week: int):
            _, x, y = self.all_data[week]
            FP, TP, FN, top_a = self.test(x, y)
            investigate_y = torch.zeros_like(y)
            investigate_y[top_a] = y[top_a]
            self.incremental_train(x, investigate_y, self.fine_tuning_epoch, weakly_target=self.weekly_target)
            n_per_class = self.K//self.n_classes
            self.exemplar_sets.reduce_size(n_per_class)
            self.create_new_class_set(x, investigate_y, n_per_class, week)
            return torch.stack([FP, TP, FN])
        stats = torch.stack([*map(test_and_tuing, self.turning_week_range)])
        FP, TP, FN = stats.sum(dim=0)
        p = TP/(TP+FP)
        r = TP/(TP+FN)
        f1 = 2*(p*r)/(p+r)
        # print(f'{self.init_lr} {self.tuning_lr} {self.weekly_target} {self.init_epoch} {self.fine_tuning_epoch}')
        print(f'{f1:.5f} {TP:.0f} {FP:.0f} {FN:.0f} {p:.5f} {r:.5f}', flush=True)
        return {'f1': f1.item(), 'TP': TP.item(), 'FP': FP.item(), 'FN': FN.item(), 'p': p.item(), 'r': r.item()}


if __name__ == '__main__':
    w, nw, e1, e2, lr1, lr2=(2, 0.0, 15, 15, 0.002, 0.002)
    trainer = IcarlTrainer(mode='day',
                           init_week_range=range(35),
                           turning_week_range=range(35, 70),
                           K=10000,
                           norm_weight=nw,
                           init_lr=lr1,
                           tuning_lr=lr2,
                           #   threhold=100,
                           weekly_target=w,
                           init_epoch=e1,
                           fine_tuning_epoch=e2
                           )
    f1 = trainer.run() 


def run_task(w: float, norm_weight: float, init_epoch: int, fine_tuning_epoch: int, init_lr: float, tuning_lr: float):
    from pathlib import Path
    from time import sleep

    tmp = Path('/tmp')
    cuda0 = tmp/'cuda0.lock'
    cuda1 = tmp/'cuda1.lock'
    using_cuda0 = True
    device = 'cuda'
    try:
        while True:
            while cuda0.exists() and cuda1.exists():
                sleep(0.1)
            try:
                if not cuda0.exists():
                    using_cuda0 = True
                    cuda0.touch(exist_ok=False)
                    break
                if not cuda1.exists():
                    using_cuda0 = False
                    cuda1.touch(exist_ok=False)
                    break
            except Exception:
                return None
        device = 'cuda:0' if using_cuda0 else 'cuda:1'
        trainer = IcarlTrainer(mode='day',
                                init_week_range=range(35),
                                init_lr=init_lr,
                                turning_week_range=range(35, 70),
                                K=10000,
                                norm_weight=norm_weight,
                                tuning_lr=tuning_lr,
                                #   threhold=1,
                                weekly_target=w*1,
                                init_epoch=init_epoch,
                                fine_tuning_epoch=fine_tuning_epoch,
                                device=device)
        result = trainer.run()
        return result
    except Exception as e:
        import sys
        print(e, file=sys.stderr, flush=True)
        return None
    finally:
        try:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            if using_cuda0:
                cuda0.unlink()
            else:
                cuda1.unlink()
        except Exception as e:
            import sys
            print(e, file=sys.stderr, flush=True)
            return None
