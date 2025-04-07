import os
import sys
import torch
from src.model import LSTMModel

def test_lstm_model():
    model = LSTMModel()
    batch_size = 32
    seq_length = 10
    input_size = 1
    
    x = torch.randn(batch_size, seq_length, input_size)
    output = model(x)
    
    assert output.shape == (batch_size, 1)
    assert isinstance(model, torch.nn.Module)
