
import argparse
import os
import numpy as np
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

def run_inference(spice_file, model_path, output_path):
    base = os.path.splitext(os.path.basename(spice_file))[0]
    current = np.loadtxt(os.path.join(args.output,f"current_map_{base}.csv"), delimiter=",")
    density = np.loadtxt(os.path.join(args.output,f"pdn_density_map_{base}.csv"), delimiter=",")
    vsource = np.loadtxt(os.path.join(args.output,f"voltage_source_map_{base}.csv"), delimiter=",")
    features = np.stack([current, density, vsource], axis=0)
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        prediction = model(input_tensor).squeeze().numpy()

    output_file = os.path.join(args.output, f"predicted_ir_drop_map_{base}.csv")
    np.savetxt(output_file, prediction, delimiter=",")
    #np.savetxt(output_path, prediction, delimiter=",")
    print(f"Prediction saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-spice_file', required=True, help="Base name of SPICE file (used for CSVs)")
    parser.add_argument('-ml_model', required=True, help="Trained model path (.pt)")
    parser.add_argument('-output', required=True, help="Output CSV file path")
    args = parser.parse_args()

    run_inference(args.spice_file, args.ml_model, args.output)
