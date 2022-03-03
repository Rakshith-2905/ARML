#!/usr/bin/env bash
python main.py --datasource=plainmulti --datadir=../SGML/data --metatrain_iterations=40000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/plainmulti/5_shot/4_vertex/data_01234 --num_filters=32 --hidden_dim=128 --emb_loss_weight=0.01
python main.py --datasource=plainmulti --datadir=xxx --metatrain_iterations=40000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=xxx --num_filters=32 --hidden_dim=128 --emb_loss_weight=0.01 --test_dataset=xxx --train=False --test_epoch=xxx