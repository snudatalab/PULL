#!/bin/bash

python main.py --data PubMed --epoch 10 --val-ratio 0.1 --test-ratio 0.1 
python main.py --data Cora_full --epoch 10 --val-ratio 0.1 --test-ratio 0.1 
python main.py --data chameleon --epoch 10 --val-ratio 0.1 --test-ratio 0.1 
python main.py --data crocodile --epoch 10 --val-ratio 0.1 --test-ratio 0.1 
python main.py --data FacebookPagePage --epoch 10 --val-ratio 0.1 --test-ratio 0.1
