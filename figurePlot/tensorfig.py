from torch.utils.tensorboard import SummaryWriter
   
train_loss = []
train_acc = []


with open("../outdata/train_result.txt", "r") as f:
    Lines = f.readlines()

    count = 0
    # 从文件中loss和acc
    for line in Lines:
        count += 1
        temp = line.split(',')
        train_loss_temp = temp[1]
        train_loss.append(float(train_loss_temp.split(':')[1]))

        train_acc_temp = temp[2]
        train_acc.append(float(train_acc_temp.split(':')[1]))   

writer = SummaryWriter(log_dir="../outdata/train_result11.txt", flush_secs=120)
for n_iter in range(len(train_acc)):
    writer.add_scalar(tag='Loss/train',
                    scalar_value=train_loss[n_iter],
                    global_step=n_iter)
    writer.add_scalar('Acc/train', train_acc[n_iter], n_iter)
writer.close()


if n_iter % self.args.log_every_n_steps == 0:
    top1, top5 = accuracy(logits, labels, topk=(1, 5))
    self.writer.add_scalar('loss', loss, global_step=n_iter)
    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)