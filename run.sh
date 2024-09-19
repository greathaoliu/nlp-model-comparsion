nohup python predict.py --year 2023 --model xlnet --cuda 0 > ./logs/2023-xlnet.txt 2>&1 &
nohup python predict.py --year 2020-2022 --model xlnet --cuda 1 > ./logs/2020-2022-xlnet.txt 2>&1 &
nohup python predict.py --year 2011-2019 --model xlnet --cuda 2 > ./logs/2011-2019-xlnet.txt 2>&1 &
nohup python predict.py --year 2023 --model roberta --cuda 3 > ./logs/2023-roberta.txt 2>&1 &
nohup python predict.py --year 2020-2022 --model roberta --cuda 0 > ./logs/2020-2022-roberta.txt 2>&1 &
nohup python predict.py --year 2011-2019 --model roberta --cuda 1 > ./logs/2011-2019-roberta.txt 2>&1 &
