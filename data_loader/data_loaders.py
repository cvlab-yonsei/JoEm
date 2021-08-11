import pickle
import torch
import numpy as np
import torch.nn.functional as F

from data_loader import PASCAL_NUM_CLASSES
from data_loader.dataset import VOCSegmentation, ContextSegmentation
from torch.utils.data import DataLoader


class VOCDataLoader():
    """
    VOC data loading demo using BaseDataLoader
    """
    def __init__(self,
                 n_unseen_classes,
                 embedding, train, val,
                 num_workers, pin_memory,
                 unseen_classes_idx, seen_classes_idx):

        self.num_classes = PASCAL_NUM_CLASSES
        self.unseen_classes_idx = unseen_classes_idx
        self.seen_classes_idx = seen_classes_idx

        self.train_set = VOCSegmentation(
            unseen_classes_idx=self.unseen_classes_idx,
            seen_classes_idx=self.seen_classes_idx,
            **train['args'],
        )
        
        self.val_set = VOCSegmentation(
            unseen_classes_idx=self.unseen_classes_idx,
            seen_classes_idx=self.seen_classes_idx,
            **val['args'],
        )

        if embedding['load_embedding']:
            normalize = False  # Default: unnormalize semantic features
            if ('normalize' in embedding.keys()) and (embedding['normalize'] is True):
                normalize = True
            embedding_dataset = 'zs3'  # Default: word2vec file from ZS3Net github repo.
            if 'embedding_dataset' in embedding.keys():
                embedding_dataset = embedding['embedding_dataset'].lower()

            self.init_embeddings(embedding["w2c_size"], embedding_dataset, normalize)

        self.init_train_kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            "batch_size": train["batch_size"],
            "shuffle": train["shuffle"],
        }

        self.init_val_kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            "batch_size": val["batch_size"],
            "shuffle": val["shuffle"],
        }

    def get_train_loader(self):
        return DataLoader(self.train_set,
                          **self.init_train_kwargs,
                          drop_last=True)

    def get_val_loader(self):
        return DataLoader(self.val_set,
                          **self.init_val_kwargs)

    def init_embeddings(self, w2c_size, embedding_dataset, normalize):
        if w2c_size == 600:  # SPNet setting
            fasttxt = self.load_obj("embeddings/context/CaGNet/fasttext")[:21]
            w2v = self.load_obj("embeddings/context/CaGNet/word2vec")[:21]

            fasttxt = torch.from_numpy(fasttxt)
            w2v = torch.from_numpy(w2v)

            fasttxt = fasttxt / fasttxt.norm(dim=1).max()  # Max normalization
            w2v = w2v / w2v.norm(dim=1).max()  # Max normalization

            fasttxt = fasttxt.numpy()
            w2v = w2v.numpy()

            embed_arr = np.concatenate((w2v, fasttxt), axis=1)  # [21, 600]

        elif w2c_size == 300:
            if embedding_dataset == 'spnet':
                embed_arr = self.load_obj("embeddings/pascal/SPNet/word2vec")  # [21, 300]
            elif embedding_dataset == 'zs3':
                # embed_arr = self.load_obj("embeddings/pascal/ZS3/norm_embed_arr_" + str(w2c_size)) # [21, 300]
                embed_arr = torch.load("embeddings/pascal/ZS3/norm_embed_arr_300_airplane.pkl").numpy()
            elif embedding_dataset == 'cagnet context':
                embed_arr = self.load_obj("embeddings/context/CaGNet/word2vec")[:21]  # [21, 300]

        self.make_embeddings(embed_arr, normalize)

    def load_obj(self, name):
        with open(name + ".pkl", "rb") as f:
            return pickle.load(f, encoding="latin-1")

    def make_embeddings(self, embed_arr, normalize):
        self.embeddings = torch.from_numpy(embed_arr)

        # L2 Noramalization
        if normalize:
            self.embeddings = F.normalize(self.embeddings, dim=-1, p=2)


class ContextDataLoader():
    """
    CONTEXT data loading demo using BaseDataLoader
    """
    def __init__(self,
                 n_unseen_classes,
                 embedding, train, val,
                 num_workers, pin_memory,
                 unseen_classes_idx, seen_classes_idx):

        self.embeddings_config = embedding
        self.unseen_classes_idx = unseen_classes_idx
        self.seen_classes_idx = seen_classes_idx

        self.train_set = ContextSegmentation(
            unseen_classes_idx=self.unseen_classes_idx,
            seen_classes_idx=self.seen_classes_idx,
            **train['args'],
        )

        self.val_set = ContextSegmentation(
            unseen_classes_idx=self.unseen_classes_idx,
            seen_classes_idx=self.seen_classes_idx,
            **val['args'],
        )

        self.num_classes = self.train_set.num_classes

        if self.embeddings_config['load_embedding']:
            self.init_embeddings(self.embeddings_config["w2c_size"])
        
        self.init_train_kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            "batch_size": train["batch_size"],
            "shuffle": train["shuffle"],
        }

        self.init_val_kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            "batch_size": val["batch_size"],
            "shuffle": val["shuffle"],
        }

    def get_train_loader(self):
        return DataLoader(self.train_set,
                          **self.init_train_kwargs,
                          drop_last=True)

    def get_val_loader(self):
        return DataLoader(self.val_set,
                          **self.init_val_kwargs)

    def init_embeddings(self, w2c_size):
        if w2c_size == 600:
            fasttxt = self.load_obj("embeddings/context/CaGNet/fasttext")
            w2v = self.load_obj("embeddings/context/CaGNet/word2vec")

            # if self.embeddings_config['normalize']:
            #     norm = np.sqrt((fasttxt**2).sum(-1)).reshape(-1, 1)
            #     fasttxt = fasttxt / norm
            #     norm = np.sqrt((w2v**2).sum(-1)).reshape(-1, 1)
            #     w2v = w2v / norm

            fasttxt = torch.from_numpy(fasttxt)
            w2v = torch.from_numpy(w2v)

            fasttxt = fasttxt / fasttxt.norm(dim=1).max()
            w2v = w2v / w2v.norm(dim=1).max()

            fasttxt = fasttxt.numpy()
            w2v = w2v.numpy()

            embed_arr = np.concatenate((w2v, fasttxt), axis=1)

        elif w2c_size == 300:
            if self.num_classes == 34:
                embed_arr = self.load_obj("embeddings/context/CaGNet/word2vec")  # SPNet Setting
            elif self.num_classes == 60:
                embed_arr = np.load("embeddings/context/ZS3/pascalcontext_class_w2c.npy")  # ZS3 Setting

            embed_arr = torch.from_numpy(embed_arr)
            embed_arr = embed_arr / embed_arr.norm(dim=1).max()
            embed_arr = embed_arr.numpy()

            # if self.embeddings_config['normalize']:
            #     norm = np.sqrt((embed_arr**2).sum(-1)).reshape(-1, 1)
            #     embed_arr = embed_arr / norm

        self.make_embeddings(embed_arr)

    def load_obj(self, name):
        with open(name + ".pkl", "rb") as f:
            return pickle.load(f, encoding="latin-1")

    def make_embeddings(self, embed_arr):
        self.embeddings = torch.from_numpy(embed_arr)
