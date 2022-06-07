import torch
import torch.nn.functional as F
import numpy as np
from itertools import combinations

'''
Cite from Ma et al.
'''

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()

class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()



def extract_embeddings(model, dataloader):
    model.eval()
    n_samples = dataloader.batch_size * len(dataloader) #所有样本
    embeddings = np.zeros((n_samples, model.n_outputs))
    labels = np.zeros(n_samples)
    k = 0

    for images, target in dataloader:
        with torch.no_grad():
            images = images.cuda()            #[36, 72, 5, 5]
            # embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()  #def get_embedding(self, x):     return self.forward(x)

            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()  #def get_embedding(self, x):     return self.forward(x)
     
            labels[k:k+len(images)] = target.numpy()
            k += len(images)

    return embeddings[0:k], labels[0:k]


def extract_embeddings01(model, dataloader):
    model.eval()
    n_samples = dataloader.batch_size * len(dataloader)  # 所有样本
    embeddings = np.zeros((n_samples, model.n_outputs))
    labels = np.zeros(n_samples)
    k = 0

    for images, target in dataloader:
        with torch.no_grad():
            images = images.cuda()  # [36, 72, 5, 5]
            # embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()  #def get_embedding(self, x):     return self.forward(x)

            embeddings[k:k + len(images)] = model.get_embedding(
                images).data.cpu().numpy()  # def get_embedding(self, x):     return self.forward(x)

            labels[k:k + len(images)] = target.numpy()
            k += len(images)

    return embeddings[0:1260], labels[0:1260]


'''熵优化'''
# entropy_loss = EntropyLoss(nn.Softmax(dim=1)(outputs[source_data.shape[0]:, :]))










