
import torch
import argparse
import os
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description='DeepRoof-2026: Weight Converter for Deployment')
    parser.add_argument('input', help='Input training checkpoint (.pth)')
    parser.add_argument('output', help='Output deployment weights (.pth)')
    return parser.parse_args()

def convert_weights():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    print(f"Loading checkpoint from {args.input}...")
    checkpoint = torch.load(args.input, map_location='cpu')
    
    # Check if this is a standard MMSEG/MMEngine checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    
    # 1. Strip 'module.' prefix (added by DistributedDataParallel)
    # 2. Skip any non-model params (optimizer/scheduler states are usually in the top-level dict, not state_dict)
    prefix = 'module.'
    for k, v in state_dict.items():
        name = k[len(prefix):] if k.startswith(prefix) else k
        new_state_dict[name] = v
        
    print(f"Strips 'module.' prefix from {len(new_state_dict)} keys.")

    # Save cleaned weights
    # We only save the state_dict to keep file size minimal (stripping optimizer, epochs, etc.)
    deploy_data = {'state_dict': new_state_dict}
    
    torch.save(deploy_data, args.output)
    
    # Analytics
    orig_size = os.path.getsize(args.input) / (1024 * 1024)
    new_size = os.path.getsize(args.output) / (1024 * 1024)
    
    print(f"Success: Cleansed weights saved to {args.output}")
    print(f"Original Size: {orig_size:.2f} MB")
    print(f"Optimized Size: {new_size:.2f} MB")
    print(f"Compression: {((orig_size - new_size) / orig_size * 100):.1f}% reduction.")

if __name__ == '__main__':
    convert_weights()
