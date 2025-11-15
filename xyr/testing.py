from dataloader import orig_train_batches

for img, target in orig_train_batches.take(1):
    print(img.shape)
    print(target.shape)
    