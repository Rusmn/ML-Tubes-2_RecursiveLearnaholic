import numpy as np

from model_implementation.layer.loss import BCELoss, CCELoss, MSELoss
from model_implementation.optimizer.optimizer import Adam, Optimizer, SGD

def _to_2d(x, input_dim):
    features = np.asarray(x, dtype=float)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.ndim != 2:
        raise ValueError
    if features.shape[1] != input_dim:
        raise ValueError
    return features

def _resolve_loss(loss_fn):
    if isinstance(loss_fn, str):
        lookup = {
            "mse": MSELoss,
            "mseloss": MSELoss,
            "bce": BCELoss,
            "bceloss": BCELoss,
            "cce": CCELoss,
            "cceloss": CCELoss,
        }
        loss_cls = lookup.get(loss_fn.lower())
        if loss_cls is None:
            raise ValueError
        return loss_cls(), loss_fn.lower()

    if callable(loss_fn):
        return loss_fn, loss_fn.__class__.__name__.lower()

    raise TypeError

def _get_opt(model, optimizer, lr, optimizer_kwargs):
    optimizer_kwargs = dict(optimizer_kwargs or {})

    if isinstance(optimizer, Optimizer):
        return optimizer

    if optimizer is None:
        return SGD(model.parameters(), lr=lr, **optimizer_kwargs)

    if isinstance(optimizer, str):
        name = optimizer.lower()
        if name == "sgd":
            return SGD(model.parameters(), lr=lr, **optimizer_kwargs)
        if name == "adam":
            return Adam(model.parameters(), lr=lr, **optimizer_kwargs)
        raise ValueError

    if isinstance(optimizer, type) and issubclass(optimizer, Optimizer):
        return optimizer(model.parameters(), lr=lr, **optimizer_kwargs)

    raise TypeError

def _encode_targets(model, y, loss_fn, fit=False):
    targets = np.asarray(y)

    if isinstance(loss_fn, BCELoss):
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.reshape(-1)
        if targets.ndim != 1:
            raise ValueError

        if fit:
            model.classes_ = np.unique(targets)
            if model.classes_.size != 2:
                raise ValueError
            model.task_ = "binary"

        mapping = {label: index for index, label in enumerate(model.classes_)}
        encoded = np.asarray([mapping[label] for label in targets], dtype=float)
        return encoded.reshape(-1, 1)

    if isinstance(loss_fn, CCELoss):
        if targets.ndim == 2:
            if fit:
                model.classes_ = np.arange(targets.shape[1])
                model.task_ = "multiclass"
            if targets.shape[1] != model.output_dim:
                raise ValueError
            return targets.astype(float)

        if targets.ndim != 1:
            raise ValueError

        if fit:
            model.classes_ = np.unique(targets)
            model.task_ = "multiclass"
            if model.classes_.size != model.output_dim:
                raise ValueError

        mapping = {label: index for index, label in enumerate(model.classes_)}
        indices = np.asarray([mapping[label] for label in targets], dtype=int)
        return np.eye(model.output_dim, dtype=float)[indices]

    targets = np.asarray(y, dtype=float)
    if targets.ndim == 1:
        if model.output_dim != 1:
            raise ValueError
        targets = targets.reshape(-1, 1)
    elif targets.ndim != 2:
        raise ValueError

    if targets.shape[1] != model.output_dim:
        raise ValueError

    if fit:
        model.classes_ = None
        model.task_ = "regression"

    return targets

def _apply_out(model, outputs, loss_fn=None):
    if model.output_activation_spec is not None:
        return outputs

    if isinstance(loss_fn, BCELoss) or model.task_ == "binary":
        return outputs.sigmoid()
    if isinstance(loss_fn, CCELoss) or model.task_ == "multiclass":
        return outputs.softmax()
    return outputs

def _reg_pen(model, regularization, regularization_strength):
    if regularization is None or regularization_strength <= 0:
        return 0.0

    regularization = regularization.lower()
    if regularization == "l1":
        return float(model.regularization_l1(regularization_strength))
    if regularization == "l2":
        return float(model.regularization_l2(regularization_strength))
    raise ValueError

def _add_reg(model, regularization, regularization_strength):
    if regularization is None or regularization_strength <= 0:
        return

    regularization = regularization.lower()
    if regularization == "l1":
        model.add_l1(regularization_strength)
        return
    if regularization == "l2":
        model.add_l2(regularization_strength)
        return
    raise ValueError

def _batches(x, y, batch_size, shuffle, rng):
    indices = np.arange(x.shape[0])
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, x.shape[0], batch_size):
        batch_indices = indices[start:start + batch_size]
        yield x[batch_indices], y[batch_indices]

def _loss_val(model, x, y, loss_fn, regularization=None, regularization_strength=0.0):
    predictions = _apply_out(model, model(x), loss_fn=loss_fn)
    targets = _encode_targets(model, y, loss_fn, fit=False)
    loss = loss_fn(predictions, targets)
    return float(loss.data) + _reg_pen(model, regularization, regularization_strength)

def _progress(epoch_index, epochs, training_loss, validation_loss):
    progress = (epoch_index + 1) / epochs
    filled = int(progress * 20)
    bar = "#" * filled + "-" * (20 - filled)
    if validation_loss is None:
        print(f"[{bar}] {epoch_index + 1}/{epochs} - loss: {training_loss:.6f}")
    else:
        print(
            f"[{bar}] {epoch_index + 1}/{epochs} - "
            f"loss: {training_loss:.6f} - val_loss: {validation_loss:.6f}"
        )

def fit_net(
    model,
    x,
    y,
    loss_fn,
    batch_size=32,
    lr=0.01,
    epochs=100,
    epoch=None,
    verbose=0,
    validation_data=None,
    optimizer=None,
    optimizer_kwargs=None,
    regularization=None,
    regularization_strength=0.0,
    shuffle=True,
):
    if epoch is not None:
        epochs = epoch

    epochs = int(epochs)
    batch_size = int(batch_size)
    lr = float(lr)

    if epochs <= 0:
        raise ValueError
    if batch_size <= 0:
        raise ValueError
    if verbose not in (0, 1):
        raise ValueError

    loss_fn, loss_name = _resolve_loss(loss_fn)
    optimizer_instance = _get_opt(model, optimizer, lr, optimizer_kwargs)

    train_x = _to_2d(x, model.input_dim)
    train_y = _encode_targets(model, y, loss_fn, fit=True)
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError

    if validation_data is not None:
        val_x_raw, val_y_raw = validation_data
        val_x = _to_2d(val_x_raw, model.input_dim)
        val_y = np.asarray(val_y_raw)
        if val_x.shape[0] != val_y.shape[0]:
            raise ValueError
    else:
        val_x = None
        val_y = None

    rng = np.random.default_rng(model.seed)
    model.loss_name_ = loss_name
    model.history_ = {"training_loss": [], "validation_loss": []}
    model.training_loss_history = model.history_["training_loss"]
    model.validation_loss_history = model.history_["validation_loss"]
    model.train()

    for epoch_index in range(epochs):
        epoch_loss_sum = 0.0
        sample_count = 0

        for batch_x, batch_y in _batches(train_x, train_y, batch_size, shuffle, rng):
            model.zero_grad()
            predictions = _apply_out(model, model(batch_x), loss_fn=loss_fn)
            loss = loss_fn(predictions, batch_y)
            objective = float(loss.data) + _reg_pen(
                model,
                regularization,
                regularization_strength,
            )

            loss.backward()
            _add_reg(model, regularization, regularization_strength)
            optimizer_instance.step()

            epoch_loss_sum += objective * batch_x.shape[0]
            sample_count += batch_x.shape[0]

        training_loss = epoch_loss_sum / sample_count
        model.history_["training_loss"].append(training_loss)

        if val_x is not None:
            validation_loss = _loss_val(
                model,
                val_x,
                val_y,
                loss_fn,
                regularization=regularization,
                regularization_strength=regularization_strength,
            )
        else:
            validation_loss = None
        model.history_["validation_loss"].append(validation_loss)

        if verbose == 1:
            _progress(epoch_index, epochs, training_loss, validation_loss)

    return model.history_

def proba(model, x):
    model.eval()
    features = _to_2d(x, model.input_dim)
    outputs = _apply_out(model, model(features))
    return np.array(outputs.data, copy=True)

def predict(model, x):
    probabilities = proba(model, x)

    if model.task_ == "binary":
        indices = (probabilities.reshape(-1) >= 0.5).astype(int)
        return model.classes_[indices]

    if model.task_ == "multiclass":
        indices = np.argmax(probabilities, axis=1)
        return model.classes_[indices]

    if probabilities.shape[1] == 1:
        return probabilities.reshape(-1)
    return probabilities
