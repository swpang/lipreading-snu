import os
import torch
from torch.utils.tensorboard import SummaryWriter
from models import lipreading
from data import DataScheduler

def train_model(
        config, model: lipreading,
        scheduler: DataScheduler,
        writer: SummaryWriter,
        resume_step: int = 0
):
    saved_model_path = os.path.join(config['log_dir'], 'ckpts')
    os.makedirs(saved_model_path, exist_ok=True)

    for step, (data, epoch) in enumerate(scheduler, resume_step):  # step starts from load_step
        # save checkpoint
        if ((step + 1) % config['ckpt_step'] == 0) or ((step + 1) % config['summary_step'] == 0):
            check_point_prefix = 'ckpt-step-'
            torch.save({
                'step': step,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'lr_scheduler_state_dict': {
                        'lr_ori': model.lr_scheduler.lr_ori,
                        'epochs': model.lr_scheduler.epochs
                    }
            },
                os.path.join(
                    saved_model_path,
                    check_point_prefix + str(step + 1)
                )
            )

            if (step + 1 - config['summary_step']) % config['ckpt_step'] != 0:
                # remove latest checkpoint
                latest_ckpt_path = os.path.join(
                    saved_model_path,
                    check_point_prefix + str(step + 1 - config['summary_step'])
                )
                if os.path.exists(latest_ckpt_path):
                    os.remove(latest_ckpt_path)

        # evaluate
        if scheduler.check_eval_step(step):
            scheduler.evaluate(model, writer, step)

        # test
        if scheduler.check_test_step(step):
            scheduler.test(model, writer, step)

        # write summary
        if scheduler.check_summary_step(step):
            model.write_summary(scheduler, step)

        train_loss = model.learn(data, step)
        # adjust lr according to cosine scheduler
        model.lr_scheduler.adjust_lr(model.optimizer, epoch)

        # model learns
        print('\r[Epoch {:4}, Step {:7}, Loss {:5}]'.format(
            epoch, step + 1, '%.4f' % train_loss), end=''
        )