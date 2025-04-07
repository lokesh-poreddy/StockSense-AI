import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.model import LSTMModel
import torch

def test_lstm_model():
    model = LSTMModel()
    batch_size = 32
    seq_length = 10
    input_size = 1
    
    # Test input shape
    x = torch.randn(batch_size, seq_length, input_size)
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, 1)
    
    # Check model parameters
    assert isinstance(model, torch.nn.Module)