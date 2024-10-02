#!/bin/bash

python -m src.clean

# Array of dropout values
num_layers_list=(3 2 1)

# Loop through dropout values
for num_layers in "${num_layers_list[@]}"
do
    # Set other parameters
    embed_dim=240
    num_heads=2
    dropout=0.3

    # Run training command
    # echo "Training with dropout $dropout"
    # python -m src.train --dropout $dropout --embed_dim $embed_dim --num_layers $num_layers --num_heads $num_heads > "$dropout-$embed_dim-$num_layers-$num_heads.txt"
    
    # Run testing command
    echo "test: model with layers $num_layers"
    python -m src.test "transformer_en_fr-$dropout-$embed_dim-$num_layers-$num_heads.pth"
    
    echo "------------------------"
done

echo "All training and testing completed."