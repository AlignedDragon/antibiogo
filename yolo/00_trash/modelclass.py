import tensorflow as tf
import keras_cv
from utils import IMG_SIZE, class_mapping, GLOBAL_CLIPNORM, LEARNING_RATE


class YOLOV8Model(tf.keras.Model):
    def __init__(
            self,
            num_classes,
            img_size,
            backbone_preset="yolo_v8_xs_backbone_coco",
            fpn_depth=1
    ):
        super().__init__()

        # Create input layer
        self.input_layer = tf.keras.layers.Input(
            shape=(img_size, img_size, 3),
            name='image_input'
        )

        # Create backbone
        self.backbone = keras_cv.models.YOLOV8Backbone.from_preset(backbone_preset)

        # Create YOLO detector
        self.detector = keras_cv.models.YOLOV8Detector(
            num_classes=num_classes,
            bounding_box_format="center_xywh",
            backbone=self.backbone,
            fpn_depth=fpn_depth
        )

        # Build model
        self.build((None, img_size, img_size, 3))

    def call(self, inputs, training=None):
        # Handle different input types
        if isinstance(inputs, tuple):
            images = inputs[0]
        else:
            images = inputs

        # Process through detector
        return self.detector(images, training=training)

    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self(images, training=True)

            # Compute loss
            loss = self.detector.compute_loss(y_true = labels, y_pred = predictions)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(labels, predictions)

        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        return results

    def test_step(self, data):
        images, labels = data

        # Forward pass
        predictions = self(images, training=False)

        # Compute loss
        loss = self.detector.compute_loss(labels, predictions)

        # Update metrics
        self.compiled_metrics.update_state(labels, predictions)

        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        return results


def create_yolo_model(
        class_mapping,
        img_size,
        learning_rate,
        global_clipnorm,
        backbone_preset="yolo_v8_xs_backbone_coco"
):
    """Create and compile YOLO model"""

    # Create model
    model = YOLOV8Model(
        num_classes=len(class_mapping),
        img_size=img_size,
        backbone_preset=backbone_preset
    )

    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        global_clipnorm=global_clipnorm
    )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=None,  # Loss is handled in train_step
        metrics=[
            keras_cv.metrics.BoxCOCOMetrics(
                bounding_box_format="center_xywh",
                name="BoxCOCO",
                evaluate_freq= 2
            )
        ]
    )

    return model


# Create model
model = create_yolo_model(
    class_mapping=class_mapping,
    img_size=IMG_SIZE,
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM
)
#
# keras_cv.callbacks.PyCOCOCallback(
#             val_ds.take(20), bounding_box_format="center_xywh"
#         )