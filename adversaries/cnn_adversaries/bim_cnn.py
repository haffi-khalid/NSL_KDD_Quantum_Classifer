# bim_cnn.py

import tensorflow as tf
import numpy as np

def generate_adversarial_examples(model, X, y, epsilon=0.05, alpha=0.01, iterations=10):
    """
    BIM attack on CNN model.
    """
    loss_object = tf.keras.losses.BinaryCrossentropy()
    adv_examples = []

    for i in range(len(X)):
        x_orig = tf.convert_to_tensor([X[i]], dtype=tf.float32)
        x_adv = tf.identity(x_orig)

        for _ in range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                prediction = model(x_adv)
                loss = loss_object(tf.convert_to_tensor([y[i]], dtype=tf.float32), prediction)

            gradient = tape.gradient(loss, x_adv)
            x_adv = x_adv + alpha * tf.sign(gradient)
            x_adv = tf.clip_by_value(x_adv, x_orig - epsilon, x_orig + epsilon)
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)  # ✅ Clip final output

        adv_examples.append(tf.squeeze(x_adv).numpy())

    return np.array(adv_examples)
