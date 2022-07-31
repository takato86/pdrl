# pdrl
private deep reinforcement learning programs

## How to use
- Run the following
```
python -m pdrl 
```
or 

```
mpiexec -n 5 python -m pdrl
```

## Optimize the hyperparameters
```
python -m pdrl -o
```

### Parallelization
```
docker run --name some-mysql -e MYSQL_ROOT_PASSWORD=test -p 3306:3306 -p 33060:33060 -d mysql:5.7
docker run --network host -it --rm mysql:5.7  mysql -h 127.0.0.1 -uroot -p -e "create database optunatest;"
```

## Check Learning
```
tensorboard --logdir=runs
```
