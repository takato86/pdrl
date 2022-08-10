import optuna


def main():
    study = optuna.load_study(
        study_name="example-study",
        storage="mysql+pymysql://root:test@localhost/optunatest"
    )
    trials = study.get_trials()
    sorted_trials = [trial for trial in trials if trial.value is not None]
    sorted_trials = sorted(sorted_trials, key=lambda x: x.value, reverse=True)
    for i, trial in enumerate(sorted_trials[:5]):
        print(f"No {i + 1}:")
        print(f"value: {trial.value}")
        print("parameters: ")
        print(trial.params)


if __name__ == "__main__":
    main()
