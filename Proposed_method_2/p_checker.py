import torch
import numpy as np

FILE = "./Proposed_method_2/best_brain_tumor_model_1010.pth"   # or your .plt file

state = torch.load(FILE, map_location="cpu")

print("Keys:")
for k in state.keys():
    print("   ", k)

# Example: inspect the FIRST conv layer
w = state["feature.4.weight"]   # adjust index to what you want
print("\nWeight shape:", w.shape)
print("Weights:\n", w.numpy())

FILE = "./Proposed_method_2/best_brain_tumor_model_1011.pth"   # or your .plt file

state = torch.load(FILE, map_location="cpu")

print("Keys:")
for k in state.keys():
    print("   ", k)

# Example: inspect the FIRST conv layer
w = state["feature.4.weight"]   # adjust index to what you want
print("\nWeight shape:", w.shape)
print("Weights:\n", w.numpy())

FILE = "./Proposed_method_2/best_brain_tumor_model_1012.pth"   # or your .plt file

state = torch.load(FILE, map_location="cpu")

print("Keys:")
for k in state.keys():
    print("   ", k)

# Example: inspect the FIRST conv layer
w = state["feature.4.weight"]   # adjust index to what you want
print("\nWeight shape:", w.shape)
print("Weights:\n", w.numpy())

FILE = "./Proposed_method_2/best_brain_tumor_model_1013.pth"   # or your .plt file

state = torch.load(FILE, map_location="cpu")

print("Keys:")
for k in state.keys():
    print("   ", k)

# Example: inspect the FIRST conv layer
w = state["feature.4.weight"]   # adjust index to what you want
print("\nWeight shape:", w.shape)
print("Weights:\n", w.numpy())