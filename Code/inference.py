import numpy as np
import json
import librosa
import torch
import yaml
from tqdm import tqdm
import os
from PIL import Image
from models import DCRN_net
import soundfile as sf

def save_sound(mel):
    mel -= 1
    mel *= 10
    mel = mel.T
    mel = np.exp(mel)
    mel -= 1.e-12
    mely = librosa.feature.inverse.mel_to_audio(
        np.array(mel, dtype=np.float32),
        sr=16000, n_fft=1024, hop_length=256, fmin=20, fmax=8000
    )
    return mely

def split_array(mel):
    k = 256
    result = []
    last_part = np.full((k,80), np.mean(mel))
    for i in range(0, len(mel), k):
        result.append(mel[i:i+k])
    last_part[:result[-1].shape[0],:result[-1].shape[1]] = result[-1]
    result[-1] = last_part
    return result

def main():
    with open('config/base/train.yaml', 'r') as input_file:
        config = yaml.safe_load(input_file)
    model = DCRN_net()
    model.load_state_dict(torch.load(config['Model']['load_model']))
    device = 'cpu'
    if config['gpu']:
        device = 'cuda:0'
        model.cuda()
    model.eval()
    with open(f"{config['Data']['metadata_dir']}/val.json", 'r') as input_file:
        data = json.load(input_file)
        with torch.no_grad():
            for path in tqdm(data['paths']):
                noisy = np.load(os.path.join(config['Data']['val_dir'], f"noisy/{path}"))
                clean = np.load(os.path.join(config['Data']['val_dir'], f"clean/{path}"))
                noisy_splited = split_array(noisy)
                pred_splited = []
                for noisy_part in noisy_splited:
                    noisy_part = noisy_part[np.newaxis, :, :]
                    noisy_part = torch.as_tensor(noisy_part, dtype=torch.float32, device=device)
                    noisy_part = noisy_part.unsqueeze(0)
                    pred_part = model(noisy_part)
                    pred_splited.append(pred_part[0][0].detach().cpu().numpy())
                predict = np.vstack(pred_splited)[:noisy.shape[0]]
                predict_wav = save_sound(predict)
                noisy_wav = save_sound(noisy)
                clean_wav = save_sound(clean)
                os.makedirs(os.path.dirname(f"{config['Model']['save_audio']}/{path}"), exist_ok=True)
                np.save(f"{config['Model']['save_audio']}/{path[:-4]}_pred.npy", predict)
                sf.write(f"{config['Model']['save_audio']}/{path[:-4]}_pred.wav", predict_wav, 16000)
                sf.write(f"{config['Model']['save_audio']}/{path[:-4]}_noisy.wav", noisy_wav, 16000)
                sf.write(f"{config['Model']['save_audio']}/{path[:-4]}_clean.wav", clean_wav, 16000)
if __name__ == '__main__':
    main()
