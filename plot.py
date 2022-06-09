import matplotlib.pyplot as plt
import pickle

with open("p1memlog.pickle", "rb") as f:
	oldSyn = pickle.load(f)
with open("p2memlog.pickle", "rb") as f:
	newSyn = pickle.load(f)
with open("p2memlogeff.pickle", "rb") as f:
	newEff = pickle.load(f)

oldSynTimes = [x / 2.0 for x in range(1, len(oldSyn))]
newSynTimes = [x / 2.0 for x in range(1, len(newSyn))]
newEffTimes = [x / 2.0 for x in range(1, len(newEff))]



plt.plot(oldSynTimes, [x / (1024 ** 2) for x in oldSyn[1:]], color='blue', label="Syntax Algorithm pre-existing implementation")
plt.plot(newSynTimes, [x / (1024 ** 2) for x in newSyn[1:]], color='red', label="Syntax Algorithm updated implementation")
plt.plot(newEffTimes, [x / (1024 ** 2) for x in newEff[1:]], color='black', label="Efficient Algorithm updated implementation")
plt.xlabel("Time (s)")
plt.ylabel("Memory (MB)")
plt.legend()
plt.title("Memory use compared between implementations")
plt.show()