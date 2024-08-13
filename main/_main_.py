'''
This file contains the runtime code with all steps necessary to  solve the underlying model.
So far implemented:
- Rep-Agent utility/consumption ratio solver using a fixed-point iteration
- Next: Implement risk-free bond pricing function.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
from keras import initializers
from datetime import datetime
import processes.autoregressiveOne as arOne
import representativeAgent.repAgent as repAgent
import helpers.gaussHermite as GHfuncs
from main.parameters import *
from helpers.utilsNN import bondPriceRf, custom_activationDebt, custom_activationStab

################ MAIN CODE ################
# Initialize exogenous processes for long-run risk and consumption of the rep Investor
xProcess = arOne.autoregressiveOne(sigX, rhoX, 0)
cProcess = arOne.autoregressiveOne(sigC, 1, xBar)
zProcess = arOne.autoregressiveOne(sigZ, rhoZ, 0)

repInvestor = repAgent.RepAgent(gamma, psi, betta)
repInvestor.setConsProcess(cProcess)
repInvestor.setLongRunRisk(xProcess)

repInvestor.initContractionSolver(10000, np.ones(10000))
repInvestor.initGaussHermite(10)

repInvestor.solveUC()

# Plot bond prices
bondPrices = np.zeros(repInvestor.xGrid.size)
for i in range(repInvestor.xGrid.size):
    bondPrices[i] = repInvestor.bondPriceRf(repInvestor.xGrid[i])

#plt.plot(repInvestor.xGrid, bondPrices)
#plt.show()

# Setup numpy Gauss-Hermite arrays
pointsXZ_, weightsXZ_ = GHfuncs.gaussHermiteMultivariate(10, [xProcess, zProcess]) # Implement correlation
pointsBP_, weightsBP_ = GHfuncs.gaussHermiteMultivariate(10, [xProcess, cProcess])


# Set up TensorFlow environment and baseline Neural net
print('tf version:', tf.__version__)
# Transform parameters to tensorflow constants
with tf.name_scope('deep_parameters'):
    GAMMA = tf.constant(gamma, dtype=tf.float32, name='gamma')
    GAMMAG = tf.constant(gammaG, dtype=tf.float32, name='gammaG')
    PSI = tf.constant(psi, dtype=tf.float32, name='psi')
    PSIG = tf.constant(psiG, dtype=tf.float32, name='psiG')
    BETTA = tf.constant(betta, dtype=tf.float32, name='betta')
    BETTAG = tf.constant(bettaG, dtype=tf.float32, name='bettaG')
    XBAR = tf.constant(xBar, dtype=tf.float32, name='xBar')
    SIGC = tf.constant(sigC, dtype=tf.float32, name='sigC')
    PHIE = tf.constant(phiE, dtype=tf.float32, name='phiE')
    RHOX = tf.constant(rhoX, dtype=tf.float32, name='rhoX')
    CORRXZ = tf.constant(corrXZ, dtype=tf.float32, name='corrXZ')
    SIGX = tf.constant(sigX, dtype=tf.float32, name='sigX')
    SIGZ = tf.constant(sigZ, dtype=tf.float32, name='sigZ')
    RHOZ = tf.constant(rhoZ, dtype=tf.float32, name='rhoZ')
    PHI = tf.constant(phi, dtype=tf.float32, name='phi')
    THETA = tf.constant(theta, dtype=tf.float32, name='theta')
    DELTA = tf.constant(delta, dtype=tf.float32, name='delta')
    RRATE = tf.constant(rRate, dtype=tf.float32, name='rRate')
    RHOT = tf.constant(rhoT, dtype=tf.float32, name='rhoT')
    KAPPA = tf.constant(kappa, dtype=tf.float32, name='kappa')
    VARPHI = tf.constant(varPhi, dtype=tf.float32, name='varPhi')
    TAUBAR = tf.constant(tauBar, dtype=tf.float32, name='tauBar')
    DEBTBAR = tf.constant(debtBar, dtype=tf.float32, name='debtBar')
    SSTAR = tf.constant(sStar, dtype=tf.float32, name='sStar')
    OUTB = tf.constant(outB, dtype=tf.float32, name='outB')
    LARGETHETA = tf.constant(largeTheta, dtype=tf.float32, name='largeTheta')
    LARGEPHI = tf.constant(largePhi, dtype=tf.float32, name='largePhi')
    DELTAI = tf.constant(deltaI, dtype=tf.float32, name='deltaI')
    LAMBDAD = tf.constant(lambdaD, dtype=tf.float32, name='lambdaD')
    STDX = tf.constant(stdX, dtype=tf.float32, name='stdX')
    STDZ = tf.constant(stdZ, dtype=tf.float32, name='stdZ')

with tf.name_scope('integration'):
    pointsXZ = tf.constant(pointsXZ_, dtype=tf.float32, name='pointsXZ')
    weightsXZ = tf.constant(weightsXZ_, dtype=tf.float32, name='weightsXZ')
    pointsBP = tf.constant(pointsBP_, dtype=tf.float32, name='pointsXZ')
    weightsBP = tf.constant(weightsBP_, dtype=tf.float32, name='weightsXZ')
    xMin = tf.constant(repInvestor.xGrid[0], dtype=tf.float32, name='xMin')
    xMax = tf.constant(repInvestor.xGrid[-1], dtype=tf.float32, name='xMin')
    ucValues = tf.constant(repInvestor.ucValues, dtype=tf.float32, name='ucValues')

# Output is debt and stab fund policy
print('##### Input Arguments #####')
lr = 1e-5 # learning rate
epochs = 400
batchSize = 128
nrOfBatches = 80
simLength = nrOfBatches * batchSize
inputDim = 6
outputDim = 3
layerNodes = [32 * 20, 32 * 20]
seed = 1 # for kernel and bias initializer (ADD TO MODEL)


# Build model (Use He initialization, Kaiming He et al. 2015)
biasInit = tf.constant_initializer(np.array([0.15, 0.005,1]))
inputs = keras.Input(shape=(inputDim,), name='input')
x1 = layers.Dense(layerNodes[0], activation='relu', kernel_initializer='he_normal', name='hidden1')(inputs)
x2 = layers.Dense(layerNodes[1], activation='relu', kernel_initializer='he_normal', name='hidden2')(x1)
#outputs = layers.Dense(outputDim, activation='sigmoid', kernel_initializer='he_normal', name='output')(x2)
output1 = layers.Dense(1, activation=custom_activationDebt, kernel_initializer='he_normal', name='outputD')(x2)
output2 = layers.Dense(1, activation=custom_activationStab, kernel_initializer='he_normal', name='outputS')(x2)
output3 = layers.Dense(1, activation='softplus', kernel_initializer='he_normal', bias_initializer='ones', name='outputV')(x2)
outputs = layers.concatenate([output1, output2, output3], name='concatAll')

#outputs={"debt": output1, "stab": output2, "value": output3}
model = keras.Model(inputs=inputs,
                    outputs=outputs)

# Instantiate the optimizer
optimizer = keras.optimizers.Adam(learning_rate=lr)
# Instantiate a loss function
loss_fn = keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error")
# Initialize the model states
debtInit = np.ones(simLength)*debtBar + np.random.normal(0, debtBar/4, size=simLength)
stabInit = np.ones(simLength)*sStar + np.random.normal(0, sStar/4, size=simLength)
tauInit = np.ones(simLength)*tauBar + np.random.normal(0, tauBar/4, size=simLength)
stateInit = np.vstack((np.ones(simLength)*debtBar + np.random.normal(0, debtBar/4, size=simLength),
                      np.ones(simLength)*sStar + np.random.normal(0, sStar/4, size=simLength),
                      np.ones(simLength)*TAUBAR + np.random.normal(0, tauBar/4, size=simLength),
                      np.random.normal(0, stdX, size=simLength),
                      np.random.normal(0, stdZ, size=simLength),
                      np.random.normal(0, sigX, size=simLength))).T

# Training loop
for epoch in range(epochs):
    print(f'Start of epoch {epoch}')

    # Simulate model to obtain dataset
    if epoch > 0 and epoch % 10 == 0:
        shocksX = np.random.normal(scale=sigX, size=simLength)
        shocksZ = np.random.normal(scale=sigZ, size=simLength)
        statesSim = np.zeros([simLength, inputDim])
        statesSim[0, :] = [debtBar, sStar, tauBar, 0, 0, 0]
        for i in range(simLength-1):
            policyNext = model(statesSim[[i], :]).numpy()
            x_next = rhoX * statesSim[i, 3] + statesSim[i, 5]
            z_next = rhoZ * statesSim[i, 4] + shocksZ[i]
            y_next = np.exp(xBar + phi * statesSim[i, 3] + statesSim[i,4])
            # Today's tax rate
            tau_next = rhoT * statesSim[i, 2] + (1 - rhoT) * tauBar + (1 - rhoT) * (kappa * (policyNext[0,0]- debtBar) + varPhi * (y_next - outB))
            statesSim[i+1,:] = np.stack([policyNext[0,0], policyNext[0,1], tau_next, x_next, z_next, shocksX[i]])

        # Normalize set
        train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(statesSim, dtype=tf.float32),
                                                            tf.zeros([simLength, outputDim])))
        train_dataset = train_dataset.shuffle(buffer_size=128).batch(batchSize)
    else:
         train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(stateInit, dtype=tf.float32),
                                                             tf.zeros([simLength, outputDim])))
         train_dataset = train_dataset.shuffle(buffer_size=128).batch(batchSize)


    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with (tf.GradientTape() as tape):
            # Forward pass
            yNext = model(x_batch_train, training=True)
            # Restandardize
            ## Start calculations ##
            # Slice prediction
            a_t = (yNext[:, 0])
            s_t = (yNext[:, 1])
            v_t = (yNext[:, 2])
            # Slice state
            a_t_1 = x_batch_train[:, 0]
            s_t_1 = x_batch_train[:, 1]
            tau_t_1 = x_batch_train[:, 2]
            x_t_1 = x_batch_train[:, 3]
            z_t = x_batch_train[:, 4]
            epsX_t = x_batch_train[:, 5]
            # Today's output
            y_t = tf.math.exp(XBAR + PHI * x_t_1 + z_t)
            # Today's tax rate
            tau_t = RHOT * tau_t_1 + (1 - RHOT) * TAUBAR + (1 - RHOT) * (
                        KAPPA * (a_t - DEBTBAR) + VARPHI * (y_t - OUTB))
            # Today's x
            x_t = RHOX * x_t_1 + epsX_t
            xAdj_t = tf.math.exp(XBAR + PHI * x_t_1)
            q_t = bondPriceRf(x_t, pointsBP, weightsBP, xMin, xMax, ucValues)
            Qs_t = tf.math.exp(tf.math.log(q_t) * DELTAI)
            # Government consumption
            savings_t = xAdj_t * a_t * q_t - a_t_1 + s_t_1 - xAdj_t * tf.math.exp(tf.math.log(q_t) * DELTAI) * s_t
            adjustmentCostB_t = (LARGETHETA / 2) * tf.pow(a_t * tf.math.exp(-z_t) - DEBTBAR, 2) * y_t
            adjustmentCostS_t = (LARGEPHI / 2) * tf.pow(s_t * tf.math.exp(-z_t) - SSTAR, 2) * y_t
            adjustmentCostB_db = LARGETHETA * (a_t * tf.math.exp(-z_t) - DEBTBAR)
            adjustmentCostS_ds = LARGEPHI * (s_t * tf.math.exp(-z_t) - SSTAR)
            g_t = tau_t * y_t + savings_t - adjustmentCostB_t - adjustmentCostS_t  # Maybe add check for maximum
            g_t = tf.where(g_t < 0, 0.0005, g_t)
            # ADD CORRECTIONS FOR GH dimensions
            z_1_t_repeat = tf.repeat(tf.expand_dims(z_t, axis=1), len(pointsXZ_), axis=1)
            z_1_t = RHOZ * z_1_t_repeat + tf.repeat(tf.transpose(tf.expand_dims(pointsXZ[:, 1], axis=1)), batchSize,
                                                    axis=0)
            epsX_1_t = tf.repeat(tf.transpose(tf.expand_dims(pointsXZ[:, 0], axis=1)), batchSize, axis=0)
            # Get Gauss-Hermite nodes over X and Z
            state_t = tf.stack([tf.reshape(tf.repeat(tf.expand_dims(a_t, axis=1), len(pointsXZ_), axis=1), [-1]),
                                tf.reshape(tf.repeat(tf.expand_dims(s_t, axis=1), len(pointsXZ_), axis=1), [-1]),
                                tf.reshape(tf.repeat(tf.expand_dims(tau_t, axis=1), len(pointsXZ_), axis=1), [-1]),
                                tf.reshape(tf.repeat(tf.expand_dims(x_t, axis=1), len(pointsXZ_), axis=1), [-1]),
                                tf.reshape(z_1_t, [-1]),
                                tf.reshape(epsX_1_t, [-1])], axis=1)
            # predict t + 1
            policy_1_t = model(state_t, training=True)
            a_1_t = (tf.reshape(policy_1_t[:, 0], [batchSize, len(pointsXZ_)]))
            s_1_t = (tf.reshape(policy_1_t[:, 1], [batchSize, len(pointsXZ_)]))
            v_1_t = (tf.reshape(policy_1_t[:, 2], [batchSize, len(pointsXZ_)]))
            # To be calculated EVG G_1_t
            weightsStackXZ = tf.repeat(tf.transpose(tf.expand_dims(weightsXZ, axis=1)), batchSize, axis=0)
            EVG_t = tf.reduce_sum(tf.multiply(tf.pow(v_1_t, tf.constant(1 - GAMMAG)), weightsStackXZ), axis=1)
            # Calculate t+1 values
            y_1_t = tf.math.exp(XBAR + PHI * tf.repeat(tf.expand_dims(x_t, axis=1), len(pointsXZ_), axis=1) + z_1_t)
            # Today's tax rate
            tau_1_t = RHOT * tf.repeat(tf.expand_dims(tau_t, axis=1), len(pointsXZ_), axis=1) + (1 - RHOT) * TAUBAR + (
                    1 - RHOT) * (
                              KAPPA * (a_1_t - DEBTBAR) + VARPHI * (y_1_t - OUTB))
            # Today's x
            x_1_t = RHOX * tf.repeat(tf.expand_dims(x_t, axis=1), len(pointsXZ_), axis=1) + epsX_1_t
            # New BondPrice
            x_1_t_transform = tf.reshape(x_1_t, [-1])
            q_1_t = bondPriceRf(x_1_t_transform, pointsBP, weightsBP, xMin, xMax, ucValues)
            q_1_t = tf.reshape(q_1_t, [batchSize, len(pointsXZ_)])
            # Government consumption
            xAdj_1_t = tf.math.exp(XBAR + PHI * x_1_t)
            savings_1_t = xAdj_1_t * a_1_t * q_1_t - tf.repeat(tf.expand_dims(a_t, axis=1), len(pointsXZ_), axis=1
                                                               ) + tf.repeat(tf.expand_dims(s_t, axis=1),
                                                                             len(pointsXZ_), axis=1
                                                                             ) - xAdj_1_t * tf.math.exp(
                tf.math.log(q_1_t) * DELTAI) * s_1_t
            adjustmentCostB_1_t = (LARGETHETA / 2) * tf.pow(a_1_t * tf.math.exp(-z_1_t) - DEBTBAR, 2) * y_1_t
            adjustmentCostS_1_t = (LARGEPHI / 2) * tf.pow(s_1_t * tf.math.exp(-z_1_t) - SSTAR, 2) * y_1_t
            g_1_t = tau_1_t * y_1_t + savings_1_t - adjustmentCostB_1_t - adjustmentCostS_1_t  # Maybe add check for maximum
            g_1_t = tf.where(g_1_t < 0, 0.0005, g_1_t)
            EVG_t_repeat = tf.repeat(tf.expand_dims(tf.pow(EVG_t, 1 / (1 - GAMMAG)), axis=1), len(pointsXZ_), axis=1)
            integrandsEG = BETTAG * tf.pow(
                tf.divide(g_1_t, tf.repeat(tf.expand_dims(g_t, axis=1), len(pointsXZ_), axis=1))
                , (-1 / PSIG)) * tf.pow(tf.divide(v_1_t, EVG_t_repeat), (1 / PSIG - GAMMAG))
            EG = tf.reduce_sum(tf.multiply(integrandsEG, weightsStackXZ), axis=1)
            # Calculate implied V_t
            PSItmp = 1 - 1 / PSIG
            vImplied_tmp = (1 - BETTAG) * tf.pow(g_t, PSItmp) + BETTAG * tf.math.exp(
                (XBAR + PHI * x_t_1) * PSItmp) * tf.pow(
                EVG_t, PSItmp / (1 - GAMMAG))
            vImplied_t = tf.pow(vImplied_tmp, 1 / PSItmp)
            # Calculate Errors
            EEA_t = tf.divide(EG + adjustmentCostB_db, q_t) - 1
            EES_t = tf.divide(EG - adjustmentCostS_ds, tf.math.exp(tf.math.log(q_t) * DELTAI)) - 1
            EEV_t = tf.divide(vImplied_t, v_t) - 1

            y_pred = tf.stack([EEA_t, EES_t, EEV_t], axis=1)
            loss_value = loss_fn(y_batch_train, y_pred)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        #loss_value = train_step(x_batch_train, y_batch_train)
        # Log every 200 batches.
        if step % 2 == 0:
            print(
                f"Training loss (for one batch) at step {step}: {float(loss_value)}"
            )










