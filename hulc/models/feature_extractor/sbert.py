from sentence_transformers import SentenceTransformer
import torch
import json
from tqdm import tqdm
model = SentenceTransformer('all-MiniLM-L12-v2')
# sentences = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.', 
#     'The quick brown fox jumps over the lazy dog.']
# sentence_embeddings = model.encode(sentences)
# print(sentence_embeddings.shape)
from pathlib import Path
path = Path('/media/nikepupu/fast/frame12update/pickup_object/')
folders = sorted(list(filter(lambda x: x.is_dir(), path.glob('*'))))
missions_json = [folder / 'mission.json' for folder in folders]
loaded_jsons = []

for mission in missions_json:
    with open(mission) as f:
        data = json.load(f)
    loaded_jsons.append(data)

for index, data in tqdm(enumerate(loaded_jsons)):
    sentence_embeddings = model.encode(data['language_annotation'])
    save_path = missions_json[index].parent / 'language_embedding.pt'
    torch.save(sentence_embeddings, save_path)
   
# print(save_path)
# print(sentence_embeddings.shape)