from abc import ABC, abstractmethod
import numpy as np


class DigitClassificationInterface(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, image):
        pass


class CNNDigitClassifier(DigitClassificationInterface):
    def __init__(self):
        try:
            self.model = tf.keras.models.load_model('cnn_model.h5')
        except:
            raise NotImplementedError('CNN classifier needs to be implemented')

    def predict(self, image):
        if image.shape != (28, 28, 1):
            raise ValueError('Invalid image shape')

        image = np.array(image).astype('float32') / 255.0

        predictions = self.model.predict(np.array([image]))[0]
        return np.argmax(predictions)


class RandomForestDigitClassifier(DigitClassificationInterface):
    def __init__(self):
        try:
            self.model = RandomForestClassifier()
        except:
            raise NotImplementedError('Random Forest classifier needs to be implemented')

    def predict(self, image):
        # Check the input shape
        if image.shape != (784,):
            raise ValueError('Invalid image shape')

        # image = image.flatten()

        predictions = self.model.predict(np.array([image]))[0]
        return predictions


class RandomDigitClassifier(DigitClassificationInterface):
    def __init__(self):
        try:
            self.model = ''
        except:
            raise NotImplementedError('Random classifier needs to be implemented')

    def predict(self, image):
        if image.shape != (10, 10):
            raise ValueError('Invalid image shape')

        return np.random.randint(10)


class DigitClassifier(CNNDigitClassifier, RandomForestDigitClassifier, RandomDigitClassifier):
    def __init__(self, algorithm):
        if algorithm == 'cnn':
            self.model = CNNDigitClassifier()
        elif algorithm == 'rf':
            self.model = RandomForestDigitClassifier()
        elif algorithm == 'rand':
            self.model = RandomDigitClassifier()
        else:
            raise ValueError('Invalid algorithm specified')

    def predict(self, image):

        return self.model.predict(image)


test = DigitClassifier('rf')
