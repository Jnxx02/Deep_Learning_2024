from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_model(self, train_data, val_data, epochs, batch_size):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(train_data, validation_data=validation_data, epochs=epochs, batch_size=batch_size)

    @abstractmethod
    def evaluate_model(self, test_data):
        return self.model.evaluate(test_data)