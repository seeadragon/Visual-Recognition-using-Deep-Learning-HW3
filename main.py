import multiprocessing
import torch
from torch import optim
from mask_rcnn import MaskRCNN


if __name__ == '__main__':
    config = {
        'num_classes': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_epochs': 50,
        'batch_size': 1,
        'learning_rate': 1e-4,
        'weight_decay': 5e-5,
        'optimizer': optim.AdamW,
        'log_dir': 'log/maskrcnn_eval',
    }
    multiprocessing.freeze_support()
    model = MaskRCNN(config)
    #model.train_model()
    model.load_model("log\maskrcnn_v6\\best_model.pth")
    model.valid()
    #model.test()
