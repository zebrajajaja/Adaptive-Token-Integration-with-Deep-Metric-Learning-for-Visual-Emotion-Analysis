import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import os

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Custom modules
from dataloader import APSEDataset
from model import APSEModel, HarmonicLoss, NearestProxyAlignmentLoss

# === Global Configuration ===
CONFIG = {
    'root_dir': "./data/project_files",
    'batch_size': 32,

    # Phase distribution
    'warmup_epochs': 3,  # Phase 1: Frozen backbones
    'total_epochs': 25,  # Total iterations

    'num_classes': 8,
    'desc_max_len': 128,  # Truncation length
    'lambda_npa': 0.35,  # Modified NPA weight
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # Visualization settings
    'tsne_max_points': 2500,
    'tsne_out_png': "analysis_output_v1.png",
    'tsne_out_pdf': None,
}


def calculate_retrieval_metrics(embeddings, labels):
    """
    Standard MPEG-7 Retrieval Metrics: RNN, FT, ST, ANMRR [cite: 1125, 1126]
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    num_samples = len(labels)
    if num_samples <= 1:
        return 0.0, 0.0, 0.0, 1.0

    # Euclidean distance matrix [cite: 1109]
    dist_matrix = torch.cdist(torch.tensor(embeddings), torch.tensor(embeddings), p=2).numpy()
    np.fill_diagonal(dist_matrix, np.inf)

    rnn_count = 0
    ft_sum = 0
    st_sum = 0
    nmrr_sum = 0
    valid_queries = 0

    for i in range(num_samples):
        query_label = labels[i]
        G = np.sum(labels == query_label) - 1
        if G <= 0:
            continue

        valid_queries += 1
        sorted_indices = np.argsort(dist_matrix[i])

        if labels[sorted_indices[0]] == query_label:
            rnn_count += 1

        top_G_indices = sorted_indices[:G]
        n_ft = np.sum(labels[top_G_indices] == query_label)
        top_2G_indices = sorted_indices[:2 * G]
        n_st = np.sum(labels[top_2G_indices] == query_label)

        ft_sum += (n_ft / G)
        st_sum += (n_st / G)

        K = 2 * G
        retrieved_indices = sorted_indices[:K]
        is_relevant = (labels[retrieved_indices] == query_label)
        correct_ranks = np.where(is_relevant)[0] + 1
        rank_sum = np.sum(correct_ranks)
        missed_count = G - len(correct_ranks)
        rank_sum += missed_count * (K + 1)
        avr = rank_sum / G
        mrr = avr - 0.5 * (1 + G)
        nmrr = mrr / (K + 0.5 - 0.5 * G)
        nmrr = max(0.0, min(1.0, nmrr))
        nmrr_sum += nmrr

    return (rnn_count / valid_queries, ft_sum / valid_queries,
            st_sum / valid_queries, nmrr_sum / valid_queries)


def train_one_epoch(model, dataloader, tokenizer, criterion_sem, criterion_npa, optimizer, epoch_idx):
    model.train()
    total_loss = 0
    stage = "WARMUP" if epoch_idx < CONFIG['warmup_epochs'] else "FINE-TUNE"
    loop = tqdm(dataloader, desc=f"Epoch {epoch_idx + 1} [{stage}]")

    for batch in loop:
        images = batch['image'].to(CONFIG['device'])
        labels = torch.tensor([CONFIG['labels_map'][l] for l in batch['label']],
                              dtype=torch.long, device=CONFIG['device'])

        desc_inputs = tokenizer(
            batch['caption'],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=CONFIG['desc_max_len']
        ).to(CONFIG['device'])

        embeddings = model(
            images=images,
            desc_input_ids=desc_inputs['input_ids'],
            desc_mask=desc_inputs['attention_mask']
        )

        loss_sem = criterion_sem(embeddings, labels)
        loss_npa = criterion_npa(embeddings, labels)
        loss = loss_sem + CONFIG['lambda_npa'] * loss_npa[cite: 1028, 1075]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Harder constraint
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def evaluate(model, dataloader, tokenizer, criterion_npa):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation Phase"):
            images = batch['image'].to(CONFIG['device'])
            labels = torch.tensor([CONFIG['labels_map'][l] for l in batch['label']],
                                  dtype=torch.long, device=CONFIG['device'])

            desc_inputs = tokenizer(
                batch['caption'],
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=CONFIG['desc_max_len']
            ).to(CONFIG['device'])

            emb = model(images, desc_inputs['input_ids'], desc_inputs['attention_mask'])
            all_embeddings.append(emb.cpu())
            all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    rnn, ft, st, anmrr = calculate_retrieval_metrics(all_embeddings, all_labels)

    # Accuracy via proxies [cite: 802, 1038]
    proxies = torch.nn.functional.normalize(criterion_npa.proxies.data.clone().squeeze(1), p=2, dim=1).to(
        CONFIG['device'])
    sim = torch.matmul(all_embeddings.to(CONFIG['device']), proxies.t())
    acc = (sim.argmax(dim=1).cpu() == all_labels).float().mean().item()

    print(f"\nMetric Summary:\nAcc: {acc:.4f} | R_NN: {rnn:.4f} | ANMRR: {anmrr:.4f}\n")
    return anmrr


def plot_tsne(features, labels, class_names, out_png):
    tsne = TSNE(n_components=2, init='pca', random_state=101, perplexity=25)
    emb2d = tsne.fit_transform(features)

    plt.figure(figsize=(7, 6))
    for cid, cname in enumerate(class_names):
        idx = (labels == cid)
        plt.scatter(emb2d[idx, 0], emb2d[idx, 1], s=5, alpha=0.7, label=cname)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    # Noise-injected data transforms 
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])

    full_ds = APSEDataset(root_dir=CONFIG['root_dir'], transform=train_transform)
    test_ds_full = APSEDataset(root_dir=CONFIG['root_dir'], transform=test_transform)

    # Obfuscated split [cite: 1095]
    train_len = int(0.75 * len(full_ds))
    val_len = len(full_ds) - train_len
    train_set, _ = random_split(full_ds, [train_len, val_len], generator=torch.Generator().manual_seed(97))
    _, test_set = random_split(test_ds_full, [train_len, val_len], generator=torch.Generator().manual_seed(97))

    # Frequency-based sampler [cite: 1102]
    train_targets = [full_ds.data_list[i]['label_idx'] for i in train_set.indices]
    class_weights = 1. / (np.bincount(train_targets) + 1)
    sampler = WeightedRandomSampler([class_weights[t] for t in train_targets], len(train_targets))

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], sampler=sampler)
    test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False)

    sorted_emotions = sorted(full_ds.emotions)
    CONFIG['labels_map'] = {label: i for i, label in enumerate(sorted_emotions)}

    model = APSEModel(text_model_name='roberta-base', visual_model_name='openai/clip-vit-base-patch32').to(
        CONFIG['device'])
    criterion_sem = HarmonicLoss(r=8.5, r1=1.5).to(CONFIG['device'])  # Parameter shift
    criterion_npa = NearestProxyAlignmentLoss(num_classes=CONFIG['num_classes']).to(CONFIG['device'])

    # === Phase 1: Warmup ===
    for p in model.text_encoder.parameters(): p.requires_grad = False
    for p in model.visual_encoder.parameters(): p.requires_grad = False

    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    for epoch in range(CONFIG['warmup_epochs']):
        train_one_epoch(model, train_loader, AutoTokenizer.from_pretrained("roberta-base"), criterion_sem,
                        criterion_npa, opt1, epoch)

    # === Phase 2: Fine-tuning === [cite: 1100, 1101]
    for p in model.parameters(): p.requires_grad = True
    opt2 = optim.AdamW([
        {'params': model.text_encoder.parameters(), 'lr': 8e-7},
        {'params': model.visual_encoder.parameters(), 'lr': 8e-7},
        {'params': model.fusion_blocks.parameters(), 'lr': 2e-5}
    ], lr=1e-5)

    scheduler = CosineAnnealingLR(opt2, T_max=15, eta_min=5e-8)
    best_anmrr = 1.0

    for epoch in range(CONFIG['warmup_epochs'], CONFIG['total_epochs']):
        train_one_epoch(model, train_loader, AutoTokenizer.from_pretrained("roberta-base"), criterion_sem,
                        criterion_npa, opt2, epoch)
        scheduler.step()
        anmrr = evaluate(model, test_loader, AutoTokenizer.from_pretrained("roberta-base"), criterion_npa)
        if anmrr < best_anmrr:
            best_anmrr = anmrr
            torch.save(model.state_dict(), "optimized_checkpoint.pth")


if __name__ == "__main__":
    main()