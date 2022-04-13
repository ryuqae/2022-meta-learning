import tensorflow as tf
import numpy as np
from importlib import reload

from model import Prototypical_Network
import model as m

reload(m)


def train_model(
    train_dataset,
    val_dataset,
    n_tasks: int,
    n_epochs: int = 20,
    n_tpe: int = 100,
    is_random: bool = False,
    N: int = 30,
):
    @tf.function
    def loss_func(support, query):
        loss, acc = model(support, query)
        return loss, acc

    @tf.function
    def train(support, query):
        # Forward & update gradients
        with tf.GradientTape() as tape:
            loss, acc = model(support, query)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Log loss and accuracy for step
        train_loss(loss)
        train_acc(acc)

    @tf.function
    def validate(support, query):
        loss, acc = loss_func(support, query)
        val_loss(loss)
        val_acc(acc)

    def on_start_epoch(epoch):
        print(f"Epoch{epoch + 1}")
        train_loss.reset_states()
        val_loss.reset_states()
        train_acc.reset_states()
        val_acc.reset_states()

    def on_end_epoch(
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        val_losses,
        val_accs,
        train_losses,
        train_accs,
    ):
        print(
            f"\t-Train Loss:{train_loss.result():.3f} | Train Acc:{train_acc.result() * 100:.2f}%\n\t-Val Loss:{val_loss.result():.3f} | Val Acc:{val_acc.result() * 100:.2f}%"
        )
        train_losses.append(train_loss.result().numpy())
        train_accs.append(train_acc.result().numpy())
        val_losses.append(val_loss.result().numpy())
        val_accs.append(val_acc.result().numpy())

    model = m.Prototypical_Network()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer)

    # Metrics to gather
    train_loss = tf.metrics.Mean(name="train_loss")
    val_loss = tf.metrics.Mean(name="val_loss")
    train_acc = tf.metrics.Mean(name="train_accuracy")
    val_acc = tf.metrics.Mean(name="val_accuracy")
    train_accs, train_losses = [], []
    val_accs, val_losses = [], []

    train_dataset.generate_task_list(n_tasks)

    for epoch in range(n_epochs):
        on_start_epoch(epoch)
        multiplier = n_tpe // n_tasks if n_tpe > n_tasks else 1
        tasks = np.repeat(np.random.permutation(range(n_tasks)), multiplier)
        for task in tasks[:n_tpe]:
            if is_random:
                support, query = train_dataset.random_data_generator(N)
            else:
                support, query = train_dataset.data_generator(task)
            val_support, val_query = val_dataset.data_generator()

            train(support, query)
            validate(val_support, val_query)
        on_end_epoch(
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_losses,
            val_accs,
            train_losses,
            train_accs,
        )

    return train_accs, train_losses, val_accs, val_losses
