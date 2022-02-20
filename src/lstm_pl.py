import pytorch_lightning as pl
import torch
import math
from torch import nn
from torch import optim
import pytorch_lightning.loggers as pl_loggers
from data_prep import Prep
from data_module import Vocab


class TextDateModule(pl.LightningDataModule):
    """Pytorch lightning data module."""

    def __init__(self, train_corpus, valid_corpus, test_corpus):
        super().__init__()
        self.batch_size = 20
        self.train = train_corpus
        self.valid = valid_corpus
        self.test = test_corpus

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train, self.batch_size, num_workers=16, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid, self.batch_size, num_workers=16, shuffle=False, drop_last=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test, self.batch_size, num_workers=16, shuffle=False, drop_last=True
        )


class TextLSTMModule(pl.LightningModule):
    """LSTM modeule."""

    def __init__(self, vocab_size):
        super().__init__()
        self.num_layers = 2
        self.hidden_size = 100  # 200
        self.embedding_size = 100
        self.vocab_size = vocab_size

        # embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # layers
        self.lstm = nn.LSTM(
            self.embedding_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.out_fc = nn.Linear(self.hidden_size, vocab_size)
        # loss funciton
        self.loss = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(0.25)

    def forward(self, data, hidden, cell):
        embedding = self.dropout(self.embedding(data))
        output, hidden = self.lstm(embedding, (hidden, cell))
        output = self.out_fc(output)
        return output.view(-1, self.vocab_size), (hidden, cell)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=5)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        hidden = torch.zeros(self.num_layers, 20, self.hidden_size).to(self.device)
        cell = torch.zeros(self.num_layers, 20, self.hidden_size).to(self.device)
        output, (hidden, cell) = self.forward(x, hidden, cell)
        loss = self.loss(output, y)
        perplexity = math.exp(loss.item())

        tensorboard_logs = {
            "perplexity": {"train": perplexity},
            "loss": {"train": loss.detach()},
        }
        self.log(
            "loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "perplexity/train",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        hidden = torch.zeros(self.num_layers, 20, self.hidden_size).to(self.device)
        cell = torch.zeros(self.num_layers, 20, self.hidden_size).to(self.device)
        output, (hidden, cell) = self.forward(x, hidden, cell)
        loss = self.loss(output, y)
        perplexity = math.exp(loss.item())

        tensorboard_logs = {
            "perplexity": {"valid": perplexity},
            "loss": {"valid": loss.detach()},
        }
        self.log(
            "loss/valid", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "perplexity/valid",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        hidden = torch.zeros(self.num_layers, 20, self.hidden_size).to(self.device)
        cell = torch.zeros(self.num_layers, 20, self.hidden_size).to(self.device)
        output, (hidden, cell) = self.forward(x, hidden, cell)
        loss = self.loss(output, y)
        perplexity = math.exp(loss.item())

        tensorboard_logs = {
            "perplexity": {"test": perplexity},
            "loss": {"test": loss.detach()},
        }
        self.log(
            "loss/test", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "perplexity/test",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "log": tensorboard_logs}

    def init_hidden(self, batch_size=20):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, cell


if __name__ == "__main__":
    p = Prep()
    # Prepare vocab
    train_corpus = p.tokenize(p.train)
    p.building_vocab(train_corpus)

    valid_corpus = p.tokenize(p.valid)
    p.building_vocab(valid_corpus)

    test_corpus = p.tokenize(p.test)
    p.building_vocab(test_corpus)

    word_freqs = p.word_freqs

    # 30 time steps.
    train = Vocab(word_freqs, train_corpus, 31)
    valid = Vocab(word_freqs, valid_corpus, 31)
    test = Vocab(word_freqs, test_corpus, 31)

    # Train LSTM
    lstm_data_module = TextDateModule(train, valid, test)
    lstm_model = TextLSTMModule(vocab_size)

    tb_logger = pl_loggers.TensorBoardLogger("./lightning_logs/", name="LSTM")
    # Define your gpu here
    trainer = pl.Trainer(logger=tb_logger, gradient_clip_val=0.5, max_epochs=20, gpus=1)
    trainer.fit(lstm_model, data_module)

    result = trainer.test(lstm_model, data_module)
    print(result)
