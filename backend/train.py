from app.modeling import load_data, train_and_save

if __name__ == "__main__":
    print(train_and_save(load_data()))
