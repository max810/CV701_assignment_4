export PYTHONPATH=/home/maksym.bekuzarov/CV701/CV701_ass4/src:$PYTHONPATH

python src/train_mpii.py \
    --arch=hg2 \
    --image-path=dataset \
    --checkpoint=checkpoint/hg2 \
    --epochs=30 \
    --train-batch=24 \
    --workers=24 \
    --test-batch=24 \
    --lr=1e-3 \
    --schedule 15 17