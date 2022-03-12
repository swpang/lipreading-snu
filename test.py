from torch.utils.tensorboard import SummaryWriter
from models.lipreading import Lipreading
from data import DataScheduler

def test_model(
        config, model: Lipreading,
        scheduler: DataScheduler,
        writer: SummaryWriter,
        step
):
   scheduler.test(model, writer, step)