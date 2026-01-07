import time
import torch
from utils import try_gpu, loss_function
from model import VGAE
from torch import nn, optim


def train_dual_VGAE(features_structure, adj_structure, features_sequence, adj_sequence, adj_label_structure,
                    adj_label_sequence, epochs, outdim_structure, outdim_sequence):
    pos_weight_structure = float(adj_label_structure.shape[0] * adj_label_structure.shape[
        0] - adj_label_structure.sum()) / adj_label_structure.sum()
    norm_structure = adj_label_structure.shape[0] * adj_label_structure.shape[0] / float(
        (adj_label_structure.shape[0] * adj_label_structure.shape[0] - adj_label_structure.sum()) * 2)

    pos_weight_sequence = float(
        adj_label_sequence.shape[0] * adj_label_sequence.shape[0] - adj_label_sequence.sum()) / adj_label_sequence.sum()
    norm_sequence = adj_label_sequence.shape[0] * adj_label_sequence.shape[0] / float(
        (adj_label_sequence.shape[0] * adj_label_sequence.shape[0] - adj_label_sequence.sum()) * 2)

    model_structure = VGAE(features_structure.shape[1], 256, outdim_structure, 0.1).to(device=try_gpu())
    model_sequence = VGAE(features_sequence.shape[1], 256, outdim_sequence, 0.1).to(device=try_gpu())

    optimizer_structure = optim.Adam(model_structure.parameters(), lr=1e-4)
    optimizer_sequence = optim.Adam(model_sequence.parameters(), lr=1e-4)

    hidden_emb_structure, hidden_emb_sequence = None, None

    for epoch in range(epochs):
        t = time.time()

        model_structure.train()
        recovered_structure, z_structure, mu_structure, logstd_structure = model_structure(features_structure,
                                                                                           adj_structure)
        loss_structure = loss_function(preds=recovered_structure, labels=adj_label_structure,
                                       mu=mu_structure, logstd=logstd_structure,
                                       norm=norm_structure, pos_weight=pos_weight_structure)

        optimizer_structure.zero_grad()
        loss_structure.backward()
        optimizer_structure.step()
        hidden_emb_structure = mu_structure

        model_sequence.train()
        recovered_sequence, z_sequence, mu_sequence, logstd_sequence = model_sequence(features_sequence, adj_sequence)
        loss_sequence = loss_function(preds=recovered_sequence, labels=adj_label_sequence,
                                      mu=mu_sequence, logstd=logstd_sequence,
                                      norm=norm_sequence, pos_weight=pos_weight_sequence)

        optimizer_sequence.zero_grad()
        loss_sequence.backward()
        optimizer_sequence.step()
        hidden_emb_sequence = mu_sequence

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch + 1:04d} | Loss (Structure): {loss_structure.item():.5f} | Loss (Sequence): {loss_sequence.item():.5f} | Time: {time.time() - t:.5f}")

    print("Optimization Finished!")

    return model_structure, hidden_emb_structure, model_sequence, hidden_emb_sequence
