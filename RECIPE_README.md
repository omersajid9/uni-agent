## Environment setup
```
bash ./setup_environment.sh
```

## Bring up ray
```
ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus=9 --num-gpus=1 --include-dashboard=true
```

## Run training
```
./start_train.sh
```