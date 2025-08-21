import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import evaluate

class LSTMLangModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super(LSTMLangModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        #Форма x - (batch_size, seq_length)

        embedded = self.embedding(x)

        lstm_out, hidden = self.lstm(embedded, hidden)

        last_hidden = lstm_out[:, -1, :]

        last_hidden = self.dropout(last_hidden)

        output = self.fc(last_hidden)

        return output, hidden
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        )
    
    def predict_next_token(self, input_tokens, vocab, device, temperature=1.0, top_k = 5):
        self.eval()
        
        with torch.no_grad():
            input_ids = []
            for token in input_tokens:
                if token in vocab:
                    input_ids.append(vocab[token])
                else:
                    input_ids.append(vocab['<UNK>'])
                    print(f"Токен '{token}' не найден в словаре, используем <UNK>")
            
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            output, _ = self(input_tensor)
            output = output / temperature
            
            probs = F.softmax(output, dim=-1)
            values, indices = torch.topk(probs, top_k, dim=-1)
            
            reverse_vocab = {v: k for k, v in vocab.items()}
            
            print("Топ-5 предсказаний:")
            for i in range(top_k):
                token_idx = indices[0][i].item()
                probability = values[0][i].item()
                token = reverse_vocab.get(token_idx, '<UNK>')
                print(f"  {i+1}. '{token}' ({token_idx}): {probability:.4f}")
            
            predicted_idx = indices[0][0].item()
            predicted_token = reverse_vocab.get(predicted_idx, '<UNK>')
        
        return predicted_token, probs.cpu().numpy()[0]

def train_model(model, train_loader, val_loader, vocab_size, device, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    rouge = evaluate.load("rouge")

    train_losses = []
    val_losses = []
    val_rouge_scores = []

    epoch_pbar = tqdm(range(num_epochs), desc="Эпохи обучения", unit="эпоха")

    for epoch in epoch_pbar:
            model.train()
            train_loss = 0
            
            train_pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs} - Обучение", leave=False)
            
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(device)
                targets = batch['target'].to(device)
                
                optimizer.zero_grad()
                
                outputs, _ = model(input_ids)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

                train_pbar.set_postfix({
                    'batch_loss': f'{loss.item():.4f}',
                    'avg_loss': f'{train_loss/len(train_pbar):.4f}'
                })
            
            train_pbar.close()
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss = 0
            all_predictions = []
            all_references = []

            val_pbar = tqdm(val_loader, desc=f"Эпоха {epoch+1}/{num_epochs} - Валидация", leave=False)

            with torch.no_grad():
                for batch in val_pbar:
                    input_ids = batch['input_ids'].to(device)
                    targets = batch['target'].to(device)
                    
                    outputs, _ = model(input_ids)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)

                    for i in range(len(predicted)):
                        pred_token_idx = predicted[i].item()
                        true_token_idx = targets[i].item()
                    
                    if true_token_idx != 0:
                        reverse_vocab = {v: k for k, v in val_loader.dataset.vocab.items()}
                        pred_token = reverse_vocab.get(pred_token_idx, '<UNK>')
                        true_token = reverse_vocab.get(true_token_idx, '<UNK>')
                        
                        all_predictions.append(pred_token)
                        all_references.append(true_token)

                    val_pbar.set_postfix({
                        'batch_loss': f'{loss.item():.4f}',
                        'avg_loss': f'{val_loss/len(val_pbar):.4f}'
                    })
            
            val_pbar.close()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            if all_predictions and all_references:
                # Для ROUGE нужно сравнивать последовательности, поэтому создаем "предложения"
                predictions_text = [' '.join(all_predictions)]
                references_text = [' '.join(all_references)]

                rouge_results = rouge.compute(
                    predictions=predictions_text, 
                    references=references_text,
                    use_stemmer=True
                )

                val_rouge_scores.append(rouge_results)
                print(f"ROUGE Scores: {rouge_results}")
            else:
                val_rouge_scores.append({})
                print("Не удалось вычислить ROUGE метрики")
            
            scheduler.step(avg_val_loss)

            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'rouge1': f'{rouge_results["rouge1"]:.4f}' if all_predictions else 'N/A',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            print(f'Эпоха {epoch+1}/{num_epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'ROUGE-1: {rouge_results["rouge1"]:.4f}, '
              f'ROUGE-2: {rouge_results["rouge2"]:.4f}, '
              f'ROUGE-L: {rouge_results["rougeL"]:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
    epoch_pbar.close()
    return train_losses, val_losses, val_rouge_scores
            