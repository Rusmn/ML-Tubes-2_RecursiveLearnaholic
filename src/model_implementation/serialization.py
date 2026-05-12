import pickle

def save_net(model, path):
    payload = {
        "config": model.config(),
        "state_dict": model.state_dict(),
        "task_": model.task_,
        "classes_": model.classes_,
        "loss_name_": model.loss_name_,
        "history_": model.history_,
    }

    with open(path, "wb") as file:
        pickle.dump(payload, file)

def load_net(path, model_class):
    with open(path, "rb") as file:
        payload = pickle.load(file)

    model = model_class(**payload["config"])
    model.load_state(payload["state_dict"])
    model.task_ = payload.get("task_")
    model.classes_ = payload.get("classes_")
    model.loss_name_ = payload.get("loss_name_")
    model.history_ = payload.get("history_", {"training_loss": [], "validation_loss": []})
    model.training_loss_history = model.history_.get("training_loss", [])
    model.validation_loss_history = model.history_.get("validation_loss", [])
    return model
