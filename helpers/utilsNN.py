'''
This file contains function useful for the calculation of the gradient in tensorflow. These functions are defined in an
external file as otherwise TF is not able to read the source-code for graph-execution
'''
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from main.parameters import *
import keras

@tf.function
def bondPriceRf(state, pointsBP, weightsBP, xMin, xMax, ucValues):
    # Declare constant from parameters
    GAMMA = tf.constant(gamma, dtype=tf.float32, name='gamma')
    PSI = tf.constant(psi, dtype=tf.float32, name='psi')
    BETTA = tf.constant(betta, dtype=tf.float32, name='betta')
    XBAR = tf.constant(xBar, dtype=tf.float32, name='xBar')
    SIGC = tf.constant(sigC, dtype=tf.float32, name='sigC')
    RHOX = tf.constant(rhoX, dtype=tf.float32, name='rhoX')

    x_t_repeat = tf.repeat(tf.expand_dims(state, axis=1), pointsBP.shape[0], axis=1)
    xUC_1_t = RHOX * x_t_repeat + tf.repeat(tf.transpose(tf.expand_dims(pointsBP[:, 0], axis=1)), state.shape[0], axis=0)
    UC_1_t = tfp.math.batch_interp_regular_1d_grid(tf.reshape(xUC_1_t, [-1]), xMin, xMax, ucValues)
    UC_1_t = tf.reshape(UC_1_t, [state.shape[0], pointsBP.shape[0]])
    weightsStack = tf.repeat(tf.transpose(tf.expand_dims(weightsBP, axis=1)), state.shape[0], axis=0)
    contUC = tf.reduce_sum(tf.multiply(weightsStack, tf.pow(UC_1_t, (1 - GAMMA))), axis=1)

    integrands1 = BETTA * tf.math.exp((-1 / PSI) * (XBAR + x_t_repeat)) * tf.math.exp(
        -GAMMA * tf.repeat(tf.transpose(tf.expand_dims(pointsBP[:, 1], axis=1)), state.shape[0], axis=0)
    ) * tf.math.exp((GAMMA - 1 / PSI) * 0.5 * (1 - GAMMA) * tf.pow(SIGC, 2))
    integrands2 = tf.pow(
        tf.divide(UC_1_t, tf.pow(tf.repeat(tf.expand_dims(contUC, axis=1), pointsBP.shape[0], axis=1), 1 / (1 - GAMMA))),
        1 / PSI - GAMMA)
    bondPrice = tf.reduce_sum(tf.multiply(weightsStack, integrands1 * integrands2), axis=1)
    return bondPrice

# Custom activation function
def custom_activationDebt(x):
    return tf.divide(0.15 * 5, 1+tf.exp(-1.5*(x-0.15)))

def custom_activationStab(x):
    return tf.divide(0.005 * 10, 1+tf.exp(-6*(x-0.005)))
