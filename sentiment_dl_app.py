import streamlit as st
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, embedding_matrix, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = False  # Set False to freeze
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        emb = self.embedding(x)
        output, (h, c) = self.lstm(emb)
        out = self.dropout(torch.cat((h[-2], h[-1]), dim=1))  # concat last hidden states
        return self.fc(out)

def tokenize(text):
    return text.split()  # already cleaned & lowercased

def text_to_indices(text, vocab):
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]

def generate_tokens_from_last(last_word, vocab, embedding_matrix, num_tokens=10):
    id_to_word = {idx: word for word, idx in vocab.items()}
    emb_matrix_torch = torch.tensor(embedding_matrix)
    emb_norm = emb_matrix_torch / emb_matrix_torch.norm(dim=1, keepdim=True)

    current_word = last_word
    generated_tokens = []

    for _ in range(num_tokens):
        word_idx = vocab.get(current_word, vocab["<unk>"])
        word_vec = emb_norm[word_idx].unsqueeze(0)

        similarity = torch.mm(word_vec, emb_norm.T).squeeze(0)
        similarity[word_idx] = -1  # avoid picking itself

        next_word_idx = torch.argmax(similarity).item()
        next_word = id_to_word[next_word_idx]

        generated_tokens.append(next_word)
        current_word = next_word

    return generated_tokens
    
# load model, vocab, embedding matrix 

with open('embedding_matrix.pkl', 'rb') as f:
  embedding_matrix = pickle.load(f)

with open('vocab_50d_freq10.pkl','rb') as v:
  vocab=pickle.load(v)

EMBED_DIM=50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiLSTMClassifier(len(vocab), EMBED_DIM, hidden_dim=128, num_classes=1,
                         embedding_matrix=embedding_matrix, pad_idx=vocab["<pad>"]).to(device)
state_dict = torch.load("dl_sentiment_model_3.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)


# Streamlit UI
st.title("Deep Sentiment Analysis App")

user_input = st.text_area("Enter text here:")

if st.button("Predict"):
    if user_input.strip():
        # Preprocess
        cleaned_text = user_input.lower().strip()

        # Convert to indices
        indices = text_to_indices(cleaned_text, vocab)
        if len(indices) == 0:
            st.write("Please enter valid text.")
        else:
            input_tensor = torch.tensor([indices], dtype=torch.long).to(device)

            # Prediction
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
                pred = 1 if prob >= 0.5 else 0

            if(pred == 1):
                pred_label = "Positive"
                st.write(f"Wow, someone is in a good mood Today!!!")
                st.write(f"**I can say this with:** {prob:.2f} confidence")
            else:
                pred_label = "Negative"
                st.write(f"Uh Oh, someone is in a bad mood Today!!!, Don't worry, things will get better.")
                st.write(f"**I can say this with:** {1 - prob:.2f} confidence")

            # Generate tokens based on last word
            embedding_tensor = torch.tensor(embedding_matrix)
            id_to_word = {idx: word for word, idx in vocab.items()}

            # Parameters
            context_window = 3  # use last N tokens to define context
            top_k = 5           # candidates for sampling
            gen_len = 10        # tokens to generate

            # Initialize hashmap with all words from original input
            used_words = {word for word in cleaned_text.split()}

            # Start with input sequence
            current_sequence = indices.copy()
            generated_tokens = []

            for _ in range(gen_len):
                # Get indices of last N tokens in the sequence
                context_indices = current_sequence[-context_window:]
                
                # Average their embeddings
                context_emb = embedding_tensor[context_indices].mean(dim=0, keepdim=True)
                
                # Find top-k+1 similar words (self might be included)
                similarities = F.cosine_similarity(context_emb, embedding_tensor)
                top_indices = similarities.topk(top_k + 1).indices.tolist()
                
                # Filter out already used words
                candidates = [idx for idx in top_indices if id_to_word[idx] not in used_words]
                
                # If no unused candidate found, pick random word from vocab
                if not candidates:
                    chosen_idx = random.randint(0, len(vocab) - 1)
                else:
                    chosen_idx = random.choice(candidates[:top_k])  # random from top-k
                
                # Save token
                chosen_word = id_to_word[chosen_idx]
                generated_tokens.append(chosen_word)
                used_words.add(chosen_word)
                
                # Append to sequence
                current_sequence.append(chosen_idx)

            # Final text
            generated_text = f"Okay, I understand the vibe I can extend your {pred_label} sentence:" + cleaned_text + " " +  " ".join(generated_tokens)

            print(f"Generated tokens: {generated_tokens}")
            print(f"Generated text: {generated_text}")

