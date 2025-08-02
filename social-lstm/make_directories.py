import os

def make_directories(base_path='.'):
    # List all directories to create relative to base_path
    dirs = [
        "log/SOCIALLSTM/GRU",
        "log/SOCIALLSTM/LSTM",
        "log/OBSTACLELSTM/GRU",
        "log/OBSTACLELSTM/LSTM",
        "log/VANILLALSTM/GRU",
        "log/VANILLALSTM/LSTM",

        "model/SOCIALLSTM/GRU",
        "model/SOCIALLSTM/LSTM",
        "model/OBSTACLELSTM/GRU",
        "model/OBSTACLELSTM/LSTM",
        "model/VANILLALSTM/GRU",
        "model/VANILLALSTM/LSTM",

        "result/SOCIALLSTM/GRU",
        "result/SOCIALLSTM/LSTM",
        "result/OBSTACLELSTM/GRU",
        "result/OBSTACLELSTM/LSTM",
        "result/VANILLALSTM/GRU",
        "result/VANILLALSTM/LSTM",

        "plot/SOCIALLSTM/GRU/plots",
        "plot/SOCIALLSTM/GRU/videos",
        "plot/SOCIALLSTM/LSTM/plots",
        "plot/SOCIALLSTM/LSTM/videos",

        "plot/OBSTACLELSTM/GRU/plots",
        "plot/OBSTACLELSTM/GRU/videos",
        "plot/OBSTACLELSTM/LSTM/plots",
        "plot/OBSTACLELSTM/LSTM/videos",

        "plot/VANILLALSTM/GRU/plots",
        "plot/VANILLALSTM/GRU/videos",
        "plot/VANILLALSTM/LSTM/plots",
        "plot/VANILLALSTM/LSTM/videos",

        "data/validation"
    ]

    for d in dirs:
        path = os.path.join(base_path, d)
        os.makedirs(path, exist_ok=True)
    print("All directories created successfully.")

if __name__ == "__main__":
    make_directories()
