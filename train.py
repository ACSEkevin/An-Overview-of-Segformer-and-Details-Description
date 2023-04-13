from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.metrics import SparseCategoricalAccuracy
from ade_dataset import *
from keras.callbacks import ModelCheckpoint
from models.structrual.u_net import UNet
import matplotlib.pyplot as plt


def main():
    model = UNet(train_gen.input_shape, classes=150, )

    # model = SegFormerB0(input_shape, num_classes=150, attention_drop_rate=0.2, drop_rate=0.1)

    model.compile(
        loss=SparseCategoricalCrossentropy(),
        optimizer=Adam(learning_rate=0.001),
        metrics=SparseCategoricalAccuracy()
    )

    checkpoint = ModelCheckpoint(filepath=f'./checkpoint/{model.name}_ade_weights.h5',
                                 monitor='val_sparse_categorical_accuracy',
                                 save_best_only=True, save_weights_only=True, mode='auto')

    print(model.summary())
    history = model.fit(train_gen, batch_size=32, epochs=20, validation_data=val_gen, callbacks=[checkpoint])
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"model test loss: {val_loss}, test accuracy: {val_acc}")

    train_acc, train_loss = history.history['sparse_categorical_accuracy'], history.history['loss']
    val_acc, val_loss = history.history['val_sparse_categorical_accuracy'], history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, color='purple')
    plt.plot(val_acc, color='red')
    plt.xlabel('$epochs$')
    plt.ylabel('$accuracy$')
    plt.legend(['train_accuracy', 'val_accuracy'])

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, color='deeppink')
    plt.plot(val_loss, color='deepskyblue')
    plt.xlabel('$epochs$')
    plt.ylabel('$loss$')
    plt.legend(['train_loss', 'val_loss'])
    plt.show()


if __name__ == '__main__':
    main()
