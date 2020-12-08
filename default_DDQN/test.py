import pickle

d = pickle.load(open("flag_ep1.pkl", "rb"))

print(sum(d) / len(d))
