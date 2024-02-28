import sys

import keras
import onnx
import tf2onnx


# Define dummy functions
def mse_loss_coords(y_true, y_pred):
    pass


def assign_metric(y_true, y_pred):
    pass


def main():

    model = keras.models.load_model(
        sys.argv[1],
        custom_objects={
            "mse_loss_coords": mse_loss_coords,
            "assign_metric": assign_metric,
        },
    )

    print(f"Loaded model from {sys.argv[1]}")
    onnx_model, _ = tf2onnx.convert.from_keras(model)

    print(f"Converted model to ONNX")
    onnx.save_model(onnx_model, sys.argv[2])

    print(f"Saved model to {sys.argv[2]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <model.h5> <output.onnx>")
        sys.exit(1)

    main()
