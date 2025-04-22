# fgsm_cnn.py

import tensorflow as tf
import numpy as np

def generate_adversarial_examples(model, X, y, epsilon=0.05):
    """
    Generate adversarial examples for CNN using FGSM.
    """
    loss_object = tf.keras.losses.BinaryCrossentropy()
    adv_examples = []

    for i in range(len(X)):
        x_tensor = tf.convert_to_tensor([X[i]], dtype=tf.float32)
        y_tensor = tf.convert_to_tensor([y[i]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            prediction = model(x_tensor)
            loss = loss_object(y_tensor, prediction)

        gradient = tape.gradient(loss, x_tensor)
        perturbation = epsilon * tf.sign(gradient)
        adv_x = x_tensor + perturbation
        adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)  # ✅ Clip to input range
        adv_examples.append(tf.squeeze(adv_x).numpy())

    return np.array(adv_examples)
