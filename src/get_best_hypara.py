import optuna


def main():
    study = optuna.load_study(
        study_name="example-study",
        storage="mysql+pymysql://root:test@localhost/optunatest"
    )
    print("Best value: {}".format(study.best_value))
    print("Best parameters:")
    print(study.best_params)


if __name__ == "__main__":
    main()
