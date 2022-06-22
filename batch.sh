for filename in ./configs/batch/*.json; do
    mpiexec -n 8 python -m pdrl --config=$filename
done
