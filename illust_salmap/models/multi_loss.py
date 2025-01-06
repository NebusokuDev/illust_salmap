from torch.nn import MSELoss, Module


class MultiInputLoss(Module):
    def __init__(self, aux_loss_weight=0.4, criterion=MSELoss()):
        super().__init__()
        self.criterion = criterion
        self.aux_loss_weight = aux_loss_weight

    def forward(self, predicts, ground_truth):
        print(len(predicts))

        main, aux = predicts
        aux_loss = self.criterion(aux, ground_truth)
        main_loss = self.criterion(main, ground_truth)
        return main_loss + self.aux_loss_weight * aux_loss
