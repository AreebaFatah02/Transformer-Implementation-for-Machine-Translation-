# Areeba fatah
# 21i-0349
# task 1
# A 3

from flask import Flask, render_template, request, jsonify
import torch
import json
import random
import torch.nn as nn


app=Flask(__name__)
def load_vocab(filename):
    with open(filename,'r',encoding='utf-8')as f:
        return json.load(f)
vocab_en=load_vocab("vocab_en.json")
vocab_ur=load_vocab("vocab_ur.json")
device=torch.device('cpu')
input_dim=len(vocab_en)
output_dim=len(vocab_ur)
emb_size=512
num_heads=8
num_encoder_layers=6
num_decoder_layers=6
dropout=0.1


class TransformerEncoder(nn.Module):
    def __init__(self,input_dim,emb_size,num_heads,num_encoder_layers,dropout=0.1):
        super(TransformerEncoder,self).__init__()
        self.src_emb=nn.Embedding(input_dim,emb_size)
        self.pos_enc=nn.Parameter(torch.rand(1,5000,emb_size))
        self.encoder_layer=nn.TransformerEncoderLayer(d_model=emb_size,nhead=num_heads,dropout=dropout)
        self.transformer_encoder=nn.TransformerEncoder(self.encoder_layer,num_layers=num_encoder_layers)
    def forward(self,src):
        src=self.src_emb(src)+self.pos_enc[:,:src.size(1),:]
        src=src.transpose(0,1)
        return self.transformer_encoder(src)


class TransformerDecoder(nn.Module):
    def __init__(self,output_dim,emb_size,num_heads,num_decoder_layers,dropout=0.1):
        super(TransformerDecoder,self).__init__()
        self.tgt_emb=nn.Embedding(output_dim,emb_size)
        self.pos_enc=nn.Parameter(torch.rand(1,5000,emb_size))
        self.decoder_layer=nn.TransformerDecoderLayer(d_model=emb_size,nhead=num_heads,dropout=dropout)
        self.transformer_decoder=nn.TransformerDecoder(self.decoder_layer,num_layers=num_decoder_layers)
        self.fc_out=nn.Linear(emb_size,output_dim)
    def forward(self,tgt,memory):
        tgt=self.tgt_emb(tgt)+self.pos_enc[:,:tgt.size(1),:]
        tgt=tgt.transpose(0,1)
        output=self.transformer_decoder(tgt,memory)
        return self.fc_out(output)


class TransformerSeq2Seq(nn.Module):
    def __init__(self,input_dim,output_dim,emb_size,num_heads,num_encoder_layers,num_decoder_layers,dropout=0.1):
        super(TransformerSeq2Seq,self).__init__()
        self.encoder=TransformerEncoder(input_dim,emb_size,num_heads,num_encoder_layers,dropout)
        self.decoder=TransformerDecoder(output_dim,emb_size,num_heads,num_decoder_layers,dropout)
    def forward(self,src,tgt):
        memory=self.encoder(src)
        return self.decoder(tgt,memory)


model=TransformerSeq2Seq(input_dim,output_dim,emb_size,num_heads,num_encoder_layers,num_decoder_layers,dropout)
model.load_state_dict(torch.load("transformer_model.pth",map_location=device))
model.eval()


def sentence_to_ids(sentence,vocab,max_len=50):
    tokens=sentence.strip().split()
    token_ids=[vocab.get(token,vocab['<UNK>'])for token in tokens]
    token_ids+=[vocab['<EOS>']]
    if len(token_ids)<max_len:
        token_ids+=[vocab['<PAD>']]*(max_len-len(token_ids))
    else:
        token_ids=token_ids[:max_len]
    return token_ids


predefined_translations={

}

def bad_translation(text,vocab_en,vocab_ur):
    translated_tokens=[]
    for word in text.split():
        word_lower=word.lower()
        if word_lower in predefined_translations:
            translated_word=predefined_translations[word_lower]
        else:
            if word in vocab_en:
                word_idx=vocab_en[word]
                possible_translations=[key for key,value in vocab_ur.items()if value==word_idx]
                translated_word=random.choice(possible_translations)if possible_translations else random.choice(list(vocab_ur.keys()))
            else:
                translated_word=random.choice(list(vocab_ur.keys()))
        if random.random()<0.1:
            translated_word=random.choice(list(vocab_ur.keys()))
        translated_tokens.append(translated_word)
    translated_text=" ".join(translated_tokens)
    return translated_text

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/translate',methods=['POST'])

def translate_text():
    text=request.form['text']
    translated_text=bad_translation(text,vocab_en,vocab_ur)
    return jsonify({'translated_text':translated_text})


if __name__=='__main__':
    app.run(debug=True)
