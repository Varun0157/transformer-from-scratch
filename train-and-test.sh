python -m src.train --dropout 0.3 --embed_dim 240 --num_layers 3 --num_heads 2 > test.txt
python -m src.test "transformer_en_fr-0.3-240-3-2.pth"
