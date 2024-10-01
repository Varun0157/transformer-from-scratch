#!/bin/bash

python -m src.clean

# Array of dropout values
dropouts=(0.1 0.3 0.5)

# Loop through dropout values
for dropout in "${dropouts[@]}"
do
    # Set other parameters
    embed_dim=240
    num_layers=3
    num_heads=2
    
    # Run training command
    echo "Training with dropout $dropout"
    python -m src.train --dropout $dropout --embed_dim $embed_dim --num_layers $num_layers --num_heads $num_heads > "$dropout-$embed_dim-$num_layers-$num_heads.txt"
    
    # Run testing command
    echo "Testing model with dropout $dropout"
    python -m src.test "transformer_en_fr-$dropout-$embed_dim-$num_layers-$num_heads.pth"
    
    echo "------------------------"
done

echo "All training and testing completed."