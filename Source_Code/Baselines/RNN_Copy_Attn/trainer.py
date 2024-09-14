import tensorflow as tf
from ShapeValidator import ShapeValidator
from RNN_Copy_Attn import RNNWithCopyAttention

class ModelTrainer:
    def __init__(self, model, optimizer, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.shape_validator = ShapeValidator()
    
    def train_step(self, input_tensor, target_tensor, hidden_state):
        with tf.GradientTape() as tape:
            predictions, hidden_state, attention_weights = self.model(input_tensor, hidden_state)
            loss = self.loss_function(target_tensor, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, hidden_state
