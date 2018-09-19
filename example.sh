TRAIN_FILE='data/conll/train-stanford-raw.conll'
TEST_FILE='data/conll/test-stanford-raw.conll'

RUN=1
NUM_EPOCHS=10
MODEL_NAME='lmparser1_mlphidden_64_l2_1e-4'
HIDDEN_UNITS_MLP=64
L2=1e-4

LM='models/hidden650_batch128_dropout0.2_lr20.0.pt'
TRAIN_DIR='data/lm/English'

python3 main.py --train $TRAIN_FILE --test $TEST_FILE --run $RUN --epochs $NUM_EPOCHS --hidden_units_mlp $HIDDEN_UNITS_MLP --l2 $L2 --model_name $MODEL_NAME --language_model $LM --train_directory $TRAIN_DIR
