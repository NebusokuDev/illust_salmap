import torch


def train_salgan(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, device):
    for real_images, real_maps in dataloader:
        real_images, real_maps = real_images.to(device), real_maps.to(device)

        # Discriminatorの訓練
        optimizer_D.zero_grad()
        fake_maps = generator(real_images)

        # 本物と偽物のラベルを作成
        real_labels = torch.ones_like(real_maps)
        fake_labels = torch.zeros_like(real_maps)

        # 本物と偽物を判別
        real_loss = criterion(discriminator(real_images, real_maps), real_labels)
        fake_loss = criterion(discriminator(real_images, fake_maps.detach()), fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Generatorの訓練
        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(real_images, fake_maps), real_labels)  # 生成器は偽物を本物に見せようとする
        g_loss.backward()
        optimizer_G.step()

    return g_loss.item(), d_loss.item()
