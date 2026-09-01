from src.model import generate_data, train
rows, labels = generate_data()
print(train(rows, labels))
