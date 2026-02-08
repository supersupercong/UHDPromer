#sleep 7h 40m 55s
python setup.py develop

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4392 basicsr/train.py -opt options/train_uhdpromer.yml --launcher pytorch