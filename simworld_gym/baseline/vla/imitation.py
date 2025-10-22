import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, get_scheduler
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm
import os

# === Config ===
LANG_MODEL = "microsoft/deberta-v3-base"
IMG_MODEL = "facebook/dinov2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (720, 600)
NUM_ACTIONS = 4

# === Dataset ===
class CityNavDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = 'TASK:' + item['subtask'] + 'HISTORY:' + item['history'] + 'ORIENTATION:' + item['orientation']
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        current_image = transforms.functional.to_tensor(item['current_view'])
        expected_image = transforms.functional.to_tensor(item['expected_view'])
        label = item['action']
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'current_image': current_image,
            'expected_image': expected_image,
            'label': torch.tensor(label, dtype=torch.long)
        }

# === Model ===
class MultimodalPolicy(nn.Module):
    def __init__(self, lang_model=LANG_MODEL, img_model=IMG_MODEL, d_model=768, nheads=8, p_drop=0.1, p_tokdrop=0.1):
        super().__init__()
        # text
        self.txt = AutoModel.from_pretrained(lang_model)
        d_txt = self.txt.config.hidden_size
        # vision (two copies)
        self.img_proc = AutoImageProcessor.from_pretrained(img_model, use_fast=True)
        self.vcur = AutoModel.from_pretrained(img_model)
        self.vexp = AutoModel.from_pretrained(img_model)
        d_vis = self.vcur.config.hidden_size
        # projections
        self.proj_txt = nn.Linear(d_txt, d_model) if d_txt != d_model else nn.Identity()
        self.proj_vis = nn.Linear(d_vis, d_model) if d_vis != d_model else nn.Identity()
        self.drop_proj = nn.Dropout(p_drop)
        # cross/self attention over concatenated tokens
        self.attn = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        # head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(p_drop),
            nn.Linear(d_model, NUM_ACTIONS)
        )
        self.p_tokdrop = p_tokdrop
        
    @staticmethod
    def masked_mean(x, mask):
        # x: [B,T,D], mask: [B,T] (1 valid, 0 pad)
        w = mask.unsqueeze(-1).type_as(x)  # [B,T,1]
        s = (x * w).sum(dim=1)             # [B,D]
        d = w.sum(dim=1).clamp_min(1e-6)   # [B,1]
        return s / d

    @torch.no_grad()
    def token_dropout(self, seq, mask_valid, p):
        if p <= 0 or not self.training:
            return seq, mask_valid
        B, T, _ = seq.size()
        drop = (torch.rand(B, T, device=seq.device) < p) & mask_valid.bool()  # [B,T,1]
        n_valid = mask_valid.sum(dim=1)
        n_drop = drop.sum(dim=1)
        fix = n_drop >= n_valid
        if fix.any():
            idx_b = torch.where(fix)[0]
            for b in idx_b.tolist():
                keepable = torch.where(mask_valid[b].bool())[0]
                k = keepable[torch.randint(len(keepable), (1,))]
                drop[b, k] = False
        seq = seq.masked_fill(drop.unsqueeze(-1), 0.0)  # [B,T,D]
        mask_valid = mask_valid & (~drop).to(mask_valid.dtype)
        return seq, mask_valid
                
    def forward(self, input_ids, attention_mask, cur_img, exp_img):
        # text sequence (no pooler)
        txt_last = self.txt(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,Tt,Dt]
        txt_tok = self.drop_proj(self.proj_txt(txt_last))  # [B,Tt,D]

        # vision patch sequences
        cur_img = self.img_proc(images=cur_img, return_tensors="pt").pixel_values.to(DEVICE)
        exp_img = self.img_proc(images=exp_img, return_tensors="pt").pixel_values.to(DEVICE)
        vcur_last = self.vcur(pixel_values=cur_img).last_hidden_state   # [B,Tc,Dv]
        vexp_last = self.vexp(pixel_values=exp_img).last_hidden_state   # [B,Te,Dv]
        vcur_tok = self.drop_proj(self.proj_vis(vcur_last))  # [B,Tc,D]
        vexp_tok = self.drop_proj(self.proj_vis(vexp_last))  # [B,Te,D]

        # concat sequences: [vision_cur | vision_exp | text]
        seq = torch.cat([vcur_tok, vexp_tok, txt_tok], dim=1)  # [B,T,D]

        # build key_padding_mask (True means pad to ignore)
        B, Tc = vcur_tok.size(0), vcur_tok.size(1)
        Te = vexp_tok.size(1)
        Tt = txt_tok.size(1)
        m_vcur = torch.ones(B, Tc, device=seq.device, dtype=attention_mask.dtype)
        m_vexp = torch.ones(B, Te, device=seq.device, dtype=attention_mask.dtype)
        m_txt  = attention_mask  # [B,Tt]
        mask_valid = torch.cat([m_vcur, m_vexp, m_txt], dim=1)          # [B,T]
        seq, mask_valid = self.token_dropout(seq, mask_valid, p=self.p_tokdrop)  # dropout tokens
        key_pad = ~mask_valid.bool()                                    # True to ignore

        # attention over all tokens
        attn_out, _ = self.attn(seq, seq, seq, key_padding_mask=key_pad)

        pooled = self.masked_mean(attn_out, mask_valid)
        return self.head(pooled)

# === Train ===
def train(model, train_loader, val_loader, epochs=50, lr=1e-3):
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=100, num_training_steps=epochs*len(train_loader))
    best_acc = float("-inf")
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            logits = model(batch['input_ids'], batch['attention_mask'], batch['current_image'], batch['expected_image'])
            loss = loss_fn(logits, batch['label'])
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    batch[k] = batch[k].to(DEVICE)
                logits = model(batch['input_ids'], batch['attention_mask'], batch['current_image'], batch['expected_image'])
                loss = loss_fn(logits, batch['label'])
                val_loss += loss.item()
                pred = logits.argmax(dim=-1)
                correct += (pred == batch['label']).sum().item()
                total += batch['label'].size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss {total_loss/len(train_loader):.4f}, Val Loss {val_loss/len(val_loader):.4f}, Acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_policy_model.pt")
            print("Best model saved.")

# === Main ===
def main():
    tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL)
    dataset = load_dataset("Jise/citynav")
    train_val = dataset['train'].train_test_split(test_size=0.1)
    train_ds = CityNavDataset(train_val['train'], tokenizer)
    val_ds = CityNavDataset(train_val['test'], tokenizer)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    model = MultimodalPolicy()
    train(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
