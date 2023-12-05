import numpy as np

# read csv from a github repo
dataset_url = list(())

# dataset_url.append("https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/raw02.csv")
dataset_url.append("https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/raw03.csv")
dataset_url.append("https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/raw05.csv")
dataset_url.append("https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/raw06.csv")
dataset_url.append("https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/raw0glide.csv")
dataset_url.append("https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/raw13.csv")
dataset_url.append("https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/raw14.csv")
# dataset_url.append("https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/test2.csv")
dataset_url.append("https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/test3.csv")
dataset_url.append("https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/test4.csv")
np.random.shuffle(dataset_url)