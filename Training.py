import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW

# Load the tokenized data from the pickle file
import pandas as pd
tokenized_data = pd.read_pickle("tokenized_data.pkl")

class TimeSeriesDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data.iloc[idx]

dataset = TimeSeriesDataset(tokenized_data)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

# Fine-tuning the T5 model
class T5FineTuner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)

trainer = pl.Trainer(max_epochs=3,
                     gpus=1,
                     benchmark=True,
                     precision=16,
                     gradient_clip_val=1.0,
                     checkpoint_callback=False,
                     progress_bar_refresh_rate=0)

model = T5FineTuner()
trainer.fit(model, train_loader)
model_path = "t5_finetuned"
model.model.save_pretrained(model_path)
model.tokenizer.save_pretrained(model_path)