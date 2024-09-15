from data_preprocessor import DataPreprocessor
from RNN_Copy_Attn import RNNWithCopyAttention
from trainer import ModelTrainer
from evaluator import ModelEvaluator
import tensorflow as tf
import pandas as pd

def load_data_from_csv(file_path):
    # Load data from the CSV file
    df = pd.read_csv(file_path)
    return df['Functional Description'].values, df['Method Name'].values

def main():
    # File paths for training and testing data
    train_file = "/mnt/data/java_train.csv"
    test_file = "/mnt/data/java_test.csv"
    tokenizer_path = "path/to/tokenizer"  # Set the path to your tokenizer model

    # Hyperparameters
    vocab_size = 10000
    embedding_dim = 256
    rnn_units = 1024
    batch_size = 64
    epochs = 10

    # Load training and testing data
    train_texts, train_labels = load_data_from_csv(train_file)
    test_texts, test_labels = load_data_from_csv(test_file)

    # Data preprocessing
    data_preprocessor = DataPreprocessor(tokenizer_path)
    train_dataset = data_preprocessor.preprocess(train_texts, batch_size)
    test_dataset = data_preprocessor.preprocess(test_texts, batch_size)

    # Model creation
    model = RNNWithCopyAttention(vocab_size, embedding_dim, rnn_units)
    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

    # Model trainer
    trainer = ModelTrainer(model, optimizer, loss_function)

    # Training loop
    for epoch in range(epochs):
        hidden_state = tf.zeros((batch_size, rnn_units))
        for input_tensor, target_tensor in train_dataset:
            loss, hidden_state = trainer.train_step(input_tensor, target_tensor, hidden_state)
            print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')

    # Model evaluator
    evaluator = ModelEvaluator(model)
    evaluator.evaluate(test_dataset, hidden_state)

if __name__ == "__main__":
    main()
