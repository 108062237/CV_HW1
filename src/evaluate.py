import torch
from sklearn.metrics import accuracy_score

def evaluate(model, dataloader, config):
    model.eval()
    results = []

    is_testset = False

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch[1][0], str):  
                is_testset = True
                images, paths = batch
                images = images.to(config['device'])

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                for path, pred in zip(paths, preds):
                    filename = path.split("/")[-1]
                    results.append((filename, pred.item()))

            else:  
                images, labels = batch
                images = images.to(config['device'])
                labels = labels.to(config['device'])

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                results.append((preds.cpu().tolist(), labels.cpu().tolist()))

    if is_testset:
        return results
    else:
        all_preds, all_labels = zip(*results)
        all_preds = [p for batch in all_preds for p in batch]
        all_labels = [l for batch in all_labels for l in batch]

        acc = accuracy_score(all_labels, all_preds)
        return acc
