from model import LanguageModel

def train_model(epochs=1000):
    model = LanguageModel()
    print("开始训练模型...")
    model.train(epochs=epochs)
    print("训练完成！")

def main():
    train_model()

if __name__ == "__main__":
    main()
